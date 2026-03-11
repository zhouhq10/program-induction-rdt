"""
dp_compressor.py — Dynamic-programming compressors for melody sequences.

This module implements three compressor variants that sit on top of the base
:class:`Compressor` (see ``base_compressor.py``).  All three share the same
rate-distortion objective.

Class hierarchy::

    Compressor  (base_compressor.py)
    └── DPCompressor
        ├── DP_PCFGCompressor
        ├── DP_AGCompressor
        └── DP_HAGCompressor
"""
import re
import os
import math


import pickle
import logging
import numpy as np
import pandas as pd

from typing import Dict, List, Tuple, Optional
from collections import deque

pd.options.mode.chained_assignment = None  # suppress SettingWithCopyWarning

from src.program.router import *
from src.domain.melody.melody_primitive import (
    MelodyProgram,
    Note,
    create_or_get_pm_from_cache,
)
from src.domain.melody.base_compressor import Compressor
from src.utils.general import *

EPS = 1e-6

# Log-probability constants for the memorise primitive:
#   p(memorise) = 0.25  (probability of choosing the memorise frame type)
#   p(note_i)   = 1/6   (uniform over 6 note values per position)
LOG_PROB_MEM = np.log(0.25) + np.log(1 / 7)
LOG_PROB_NOTE = np.log(1 / 6)


# ---------------------------------------------------------------------------
# Base DP compressor
# ---------------------------------------------------------------------------


class DPCompressor(Compressor):
    """Base class for DP-based melody compressors.

    Extends :class:`Compressor` with:
    - Frame generation strategies (on-the-fly, pre-sampled, or from file).
    - A ``run_per_subtask`` method that selects the best single-program
      explanation for a contiguous sub-melody segment.
    """

    def __init__(self, program_lib: object, args: object) -> None:
        super().__init__(program_lib, args)

        self.submelody_backtrack_budget = args.submelody_backtrack_budget
        self.lossless = args.lossless
        self.frame_gen = args.frame_gen

        # Snapshot of the production table before any library updates
        self.init_pm = self.lib.production

        self.reuse_num_note_subtask = args.reuse_num_note_subtask

        self.curriculum = args.curriculum
        self.continue_infer = args.continue_infer

        # Frame sampling strategy: "switch" probabilistically alternates
        # between depth-1 and deeper frames; "uniform" samples uniformly.
        self.sample_subprog = "switch"
        self.switch = math.exp(-1) / (math.exp(-1) + math.exp(-2))

    # ------------------------------------------------------------------
    # Program reconstruction helpers
    # ------------------------------------------------------------------

    @staticmethod
    @np.vectorize
    def _reconstruct_subtask(cl_prog_term: str) -> np.ndarray:
        """Execute a program string and return the reconstructed note sequence.

        Args:
            cl_prog_term: String representation of a :class:`MelodyProgram`,
                e.g. ``"[B,repeat,note_1_2_3]"``.

        Returns:
            Numpy array of reconstructed note values.
        """
        cur_prog = MelodyProgram(cl_prog_term)
        return cur_prog.run()

    @staticmethod
    def _comp_subprog_recon_len(cur_all_progs: pd.DataFrame) -> pd.DataFrame:
        """Evaluate all program terms and record their reconstruction lengths.
        Args:
            cur_all_progs: DataFrame of candidate programs with a ``"term"``
                column.

        Returns:
            The same DataFrame with ``"recon"`` and ``"recon_len"`` added.
        """
        cur_all_progs["recon"] = cur_all_progs["term"].apply(
            DPCompressor._reconstruct_subtask
        )
        cur_all_progs["recon_len"] = cur_all_progs["recon"].apply(len)
        return cur_all_progs

    # ------------------------------------------------------------------
    # Memorise-frame construction
    # ------------------------------------------------------------------

    def add_frame_for_mem_constrained_by_len(self, subtask: np.ndarray) -> pd.DataFrame:
        """Build a memorise-program frame that exactly covers the full subtask.

        Args:
            subtask: 1-D integer array representing the sub-melody to memorise.

        Returns:
            Single-row DataFrame containing the memorise program, its
            log-probability, reconstruction, and distortion (0).
        """
        num_note = len(subtask)

        note_name = Note.array_to_string(subtask)
        possible_notes = create_or_get_pm_from_cache(note_name)

        log_prog_whole = LOG_PROB_MEM + LOG_PROB_NOTE * num_note
        memorized_progs = pd.DataFrame(
            {
                "term": f"[K,memorize,{possible_notes.name}]",
                "comp_lp": log_prog_whole,
                "log_prob": log_prog_whole,
                "type_string": "note->note",
                "distortion": 0,
                "recon": [np.array(possible_notes.value)],
                "recon_len": num_note,
                "frame": "[K,memorize,note]",
                "depth": 1,
            },
            index=[0],
        )

        memorized_progs = self._comp_ll_value(memorized_progs)
        return memorized_progs

    # ------------------------------------------------------------------
    # Per-subtask selection (inner step of the DP)
    # ------------------------------------------------------------------

    def run_per_subtask(
        self,
        subtask: np.ndarray,
        index: List[int] = [0, 0],
    ) -> pd.DataFrame:
        """Select the best single program for one sub-melody segment.

        This is the innermost computation unit of the DP.  For a given
        contiguous sub-melody ``subtask`` it:

        1. Builds a memorise frame that covers the full segment exactly.
        2. Generates ``search_budget`` additional program frames (via
           ``fill_frame``).
        3. Filters all frames to those whose reconstruction length equals
           ``len(subtask)``.
        4. Computes the rate-distortion value for each surviving frame.
        5. Returns the frame with the highest value.

        Args:
            subtask: 1-D integer array of the sub-melody to compress.
            index: ``[task_index, subtask_index]`` — used only when
                ``frame_gen == "extract"`` to locate the pre-generated frame
                file.

        Returns:
            Single-row DataFrame of the selected (highest-value) program.
        """
        # Build a zero-distortion memorise baseline
        mem_prog = self.add_frame_for_mem_constrained_by_len(subtask)

        # Sample / generate candidate program frames
        filled = self.fill_frame(subtask)

        # Combine memorise frame with sampled frames
        filled = pd.concat([filled, mem_prog], ignore_index=True)

        # Keep only programs whose reconstruction length matches the subtask
        filled = self._comp_subprog_recon_len(filled)
        filled = filled[filled["recon_len"] == len(subtask)]

        # Compute rate-distortion value for each surviving program
        filled = self._comp_subprog_value(filled, subtask)

        # Optionally enforce zero-distortion (lossless mode)
        if self.lossless:
            filled = filled[filled["log_ll"] > 0]

        # Select the program with the highest value
        chosen_subprog = filled.loc[[filled["value"].idxmax()]]
        return chosen_subprog

    # ------------------------------------------------------------------
    # Frame generation strategies
    # ------------------------------------------------------------------

    def _generate_frames_on_the_fly(self) -> pd.DataFrame:
        """Generate ``search_budget`` program frames by sampling the grammar.

        Samples a type signature and program depth for each frame, then calls
        :meth:`Grammar.generate_frame` to produce a concrete program.

        Returns:
            DataFrame of ``search_budget`` generated frames.
        """
        filled_progs = []
        for _ in range(self.search_budget):
            type_string = self.sample_type_string_subtask(num_sample=1)[0]
            depth = self.sample_depth_subtask(num_sample=1)[0]
            filled_prog = self.lib.generate_frame(type_string, depth)
            filled_progs.append(filled_prog)
        return pd.concat(filled_progs, ignore_index=True)

    def _generate_frames_by_sampling(self, subtask: np.ndarray) -> pd.DataFrame:
        """Repeatedly sample frames until one matches the subtask length.

        Args:
            subtask: Target sub-melody; used only to check reconstruction
                length.

        Returns:
            Single-row DataFrame of the first length-matching frame found.
        """
        while True:
            type_string = self.sample_type_string_subtask(num_sample=1)[0]
            depth = self.sample_depth_subtask(num_sample=1)[0]
            filled_prog = self.lib.unfold_prog_with_lp(
                type_string, depth, self.frame_sample
            )

            cl_prog_term = filled_prog["term"][0]
            cur_prog_recon_len = len(MelodyProgram(cl_prog_term).run())

            if cur_prog_recon_len == len(subtask):
                return filled_prog

    def _extract_frames_from_file(self, index: List[int]) -> pd.DataFrame:
        """Load pre-generated frames from disk for a specific task and subtask.

        Args:
            index: ``[task_index, subtask_index]`` identifying which pre-
                generated frame file to load.

        Returns:
            DataFrame of ``search_budget`` filled programs.
        """
        frame_path = (
            f"{self.frame_folder}/search_budget_{self.search_budget}_0"
            f"/task_{index[0]}/index_{index[1]}.csv"
        )
        sampled_frames = pd.read_csv(frame_path, index_col=0)

        filled_progs = [
            self._fill_single_frame(sampled_frames.iloc[i])
            for i in range(self.search_budget)
        ]
        return pd.concat(filled_progs, ignore_index=True)

    def fill_frame(
        self,
        subtask: np.ndarray,
        index: List[int] = [0, 0],
    ) -> pd.DataFrame:
        """Dispatch to the appropriate frame generation strategy.

        Selects among ``"fly"``, ``"sample"``, and ``"extract"`` based on
        ``self.frame_gen``.

        Args:
            subtask: Target sub-melody (used by the ``"sample"`` strategy to
                enforce length matching).
            index: Task / subtask index pair (used by the ``"extract"``
                strategy).

        Returns:
            DataFrame of candidate program frames.
        """
        if self.frame_gen == "fly":
            return self._generate_frames_on_the_fly()
        elif self.frame_gen == "sample":
            return self._generate_frames_by_sampling(subtask)
        elif self.frame_gen == "extract":
            return self._extract_frames_from_file(index)
        else:
            raise ValueError(f"Invalid frame generation method {self.frame_gen}")


# ---------------------------------------------------------------------------
# PCFG compressor  (no library update)
# ---------------------------------------------------------------------------


class DP_PCFGCompressor(DPCompressor):
    """DP compressor with a fixed PCFG — no library update across tasks.

    The grammar prior is fixed at the uniform PCFG initialised in
    ``0_construct_pm.py``.  Each melody is segmented independently into
    sub-melodies; the segmentation that minimises the cumulative RD cost.
    """

    def __init__(self, program_lib: object, args: object) -> None:
        super().__init__(program_lib, args)
        # Path to pre-generated frame files on the compute cluster.
        # NOTE: update this path if running outside the original cluster.
        self.frame_folder = ""

    def _fill_single_frame(self, sampled_frame: pd.Series) -> pd.DataFrame:
        """Instantiate a pre-sampled frame by filling in concrete primitives.

        Takes a frame skeleton (type signature + combinator structure) and
        calls :meth:`Grammar.unfold_frame` to substitute actual primitive
        terms, yielding a runnable program.

        Args:
            sampled_frame: A single row from a pre-generated frame CSV,
                containing at least ``"term"``, ``"type_string"``,
                ``"frame"``, and ``"depth"`` columns.

        Returns:
            DataFrame of filled programs derived from this frame.
        """
        filled_prog = self.lib.unfold_frame(
            sampled_frame["term"], sampled_frame["type_string"]
        )
        filled_prog["frame"] = sampled_frame["frame"]
        filled_prog["depth"] = sampled_frame["depth"]
        return filled_prog

    def _find_best_submelody(
        self,
        task: np.ndarray,
        index: int,
        rates: deque,
        distortions: deque,
        prog_list: deque,
    ) -> Tuple[float, float, float, pd.DataFrame]:
        """Inner DP step: find the optimal sub-melody ending at ``index``.

        Iterates over all possible sub-melody lengths ``l`` (1 to
        ``submelody_backtrack_budget``) and selects the one that minimises

            cost(l) = β · (R_l + R_prev)  +  (D_l + D_prev)

        where ``R_prev`` / ``D_prev`` are the cumulative rate / distortion up
        to ``index - l``, stored in the rolling deques.

        Args:
            task: Full melody as a 1-D integer array.
            index: Current note position (1-indexed) in the melody.
            rates: Deque of cumulative rates; ``rates[-l]`` gives the rate
                accumulated up to note ``index - l``.
            distortions: Deque of cumulative distortions (same indexing).
            prog_list: Deque of program DataFrames accumulated so far.

        Returns:
            Tuple ``(min_cost, best_rate, best_dist, best_prog)``:
            - ``min_cost``: Minimum total RD cost achievable at ``index``.
            - ``best_rate``: Cumulative rate for the optimal segmentation.
            - ``best_dist``: Cumulative distortion for the optimal segmentation.
            - ``best_prog``: DataFrame of all selected programs up to ``index``.
        """
        min_cost = float("inf")
        best_rate, best_dist, best_prog = None, None, None

        for l in range(1, min(self.submelody_backtrack_budget, index) + 1):
            subtask = task[index - l : index]
            cur_prog = self.run_per_subtask(subtask)

            cur_rate = -cur_prog["log_prob"].item()
            cur_dist = cur_prog["distortion"].item()

            # Total RD cost: β · (current + past rate) + (current + past distortion)
            total_cost = self.beta * (cur_rate + rates[-l]) + (
                cur_dist + distortions[-l]
            )

            if total_cost < min_cost:
                min_cost = total_cost
                best_rate = cur_rate + rates[-l]
                best_dist = cur_dist + distortions[-l]
                best_prog = pd.concat([prog_list[-l], cur_prog], ignore_index=True)

        return min_cost, best_rate, best_dist, best_prog

    def run_per_task(
        self, task: np.ndarray, task_ind: int = 0
    ) -> Tuple[float, float, pd.DataFrame]:
        """Run the inner DP over one full melody.

        Segments the melody into sub-melodies by solving the 1-D DP.

        where W = ``submelody_backtrack_budget``.  Uses rolling :class:`deque`
        objects of length W+1 to avoid storing the full DP table.

        Args:
            task: Full melody as a 1-D integer array of note values.
            task_ind: Task index (used only for logging).

        Returns:
            Tuple ``(rate, distortion, prog_list)`` for the optimal
            segmentation of this melody:
            - ``rate``: Total negative log-probability of selected programs.
            - ``distortion``: Total Levenshtein distortion.
            - ``prog_list``: DataFrame of all selected programs.
        """
        n = len(task)
        note_window = self.submelody_backtrack_budget

        # Rolling deques of length note_window + 1 store the DP values for
        # the last note_window positions (+ a zero-initialised base case).
        rates = self._construct_deque(note_window + 1, 0)
        distortions = self._construct_deque(note_window + 1, 0)
        prog_list = self._construct_deque(note_window + 1, pd.DataFrame())

        for i in range(1, n + 1):
            print(f"Processing index {i}/{n}", flush=True)

            _, min_rate, min_dist, new_prog = self._find_best_submelody(
                task, i, rates, distortions, prog_list
            )

            rates.append(min_rate)
            distortions.append(min_dist)
            prog_list.append(new_prog)

        # The last entry in each deque holds the optimal value for the full melody
        return rates[-1], distortions[-1], prog_list[-1]

    def run(
        self,
        tasks: List[np.ndarray],
        save_path: str,
    ) -> None:
        """Compress all tasks and save results to disk.

        Iterates over tasks ``[task_start_ind, task_start_ind + task_num)``,
        calls :meth:`run_per_task` on each, and saves the selected program
        trajectory as a pickle.

        Args:
            tasks: List of melody arrays (loaded from the train task pickle).
            save_path: Directory path where per-task ``.obj`` files are saved.
        """
        for n in range(self.task_start_ind, self.task_start_ind + self.task_num):
            cur_task = tasks[n]
            print(f"Task {n}/{len(tasks)}", flush=True)

            _, _, prog_trajs = self.run_per_task(cur_task)

            if save_path:
                save_dict = {"prog_trajs": prog_trajs}
                self.save_result_per_task(n, save_dict, save_path)


# ---------------------------------------------------------------------------
# Adaptor Grammar compressor  (global library updated across tasks)
# ---------------------------------------------------------------------------


class DP_AGCompressor(DPCompressor):
    """DP compressor with an Adaptor Grammar (AG) global library.

    After processing each melody the global program library is updated via
    a Pitman-Yor posterior update (:meth:`AdaGrammar.update_post_lib`).  An
    outer DP over a sliding window of ``max_melody_num`` past melodies
    determines which library state (from which past melody to start
    re-processing) minimises the cumulative RD cost.

    The outer DP works as follows: for each new melody at position ``n``,
    the algorithm tries all ``l`` ∈ {0, …, max_melody_num} and re-processes
    the block of melodies ``[n-l, …, n]`` starting from the library state
    that was active before melody ``n-l``.  The block with the lowest total
    cost is kept.
    """

    def __init__(self, program_lib: object, args: object) -> None:
        super().__init__(program_lib, args)
        self.max_melody_num = args.max_melody_num

    def run_inner_dp_per_task(
        self,
        task: np.ndarray,
    ) -> Tuple[float, float, pd.DataFrame]:
        """Run the PCFG inner DP for one melody (delegates to DP_PCFGCompressor).

        This method exists so that :class:`DP_HAGCompressor` can override it
        with its local-library variant without duplicating the outer DP code.

        Args:
            task: Melody array.

        Returns:
            ``(rate, distortion, prog_list)`` from the inner DP.
        """
        return DP_PCFGCompressor.run_per_task(self, task)

    def _find_best_submelody(
        self,
        task: np.ndarray,
        index: int,
        rates: deque,
        distortions: deque,
        prog_list: deque,
    ) -> Tuple[float, float, float, pd.DataFrame]:
        """Delegate inner-DP sub-melody selection to the PCFG implementation."""
        return DP_PCFGCompressor._find_best_submelody(
            self, task, index, rates, distortions, prog_list
        )

    def run_per_task(
        self,
        tasks: List[np.ndarray],
        cur_task_ind: int,
        cache: List,
    ) -> Tuple[List, List, List, List]:
        """Outer DP step: find the optimal melody-level backtracking for task ``n``.

        Tries all ``l`` ∈ {0, …, min(max_melody_num, cur_task_ind)} and
        re-processes the block ``tasks[n-l : n+1]`` starting from the cached
        library state that preceded melody ``n-l``.  Keeps the block whose
        total cost (sum of distortions + β · sum of rates) is smallest.

        Args:
            tasks: Full list of melody arrays.
            cur_task_ind: Index of the current melody (``n``).
            cache: List of four deques ``[rates, distortions, progs, libs]``
                maintained across melody iterations.

        Returns:
            Tuple of four lists (sliced to the last ``max_melody_num+1``
            elements) for the optimal block:
            ``(rates, distortions, prog_trajectories, library_states)``.
        """
        max_melody_num = self.max_melody_num
        cache_rates, cache_distortions, cache_progs, cache_global_lib = cache

        min_cost = float("inf")
        min_rate, min_distortion, min_prog_traj, min_global_lib = [], [], [], []

        for l in range(0, min(max_melody_num, cur_task_ind) + 1):
            temp_ind = -(l + 1)
            # Retrieve the cached state from l melodies ago
            temp_progs = list(cache_progs[temp_ind])
            temp_libs = list(cache_global_lib[temp_ind])
            temp_rates = list(cache_rates[temp_ind])
            temp_distortions = list(cache_distortions[temp_ind])

            # Restore the library to the state before melody n-l
            self.lib.production = temp_libs[-1]

            # Re-process the block [n-l, …, n] with this library state
            for n in range(cur_task_ind - l, cur_task_ind + 1):
                task = tasks[n]

                cur_rate, cur_dist, cur_prog = self.run_inner_dp_per_task(task)

                temp_progs.append(cur_prog)
                temp_rates.append(cur_rate)
                temp_distortions.append(cur_dist)

                # Update the global library after each melody in the block
                self.lib.update_post_lib(cur_prog)
                temp_libs.append(self.lib.production)

            cost = sum(temp_distortions) + self.beta * sum(temp_rates)

            if cost < min_cost:
                min_cost = cost
                min_rate = temp_rates
                min_distortion = temp_distortions
                min_prog_traj = temp_progs
                min_global_lib = temp_libs

        # Keep only the most recent max_melody_num+1 entries (sliding window)
        slice_start = -max_melody_num - 1
        return (
            min_rate[slice_start:],
            min_distortion[slice_start:],
            min_prog_traj[slice_start:],
            min_global_lib[slice_start:],
        )

    def run(
        self,
        tasks: List[np.ndarray],
        save_path: str,
    ) -> None:
        """Compress all tasks with AG library updates and save results.

        Maintains rolling deques of length ``max_melody_num + 1`` for rates,
        distortions, program trajectories, and library states.  Supports
        resuming an interrupted run via ``--continue_infer``.

        Args:
            tasks: List of melody arrays.
            save_path: Output directory for per-task pickle files.
        """
        max_len = self.max_melody_num + 1

        if self.continue_infer:
            # Resume: find the last saved task index and reload cached deques
            prog_names = [
                file_name for file_name in os.listdir(save_path) if "prog" in file_name
            ]
            indexs = [
                int(re.search(r"task_(\d+)_", file_name).group(1))
                for file_name in prog_names
            ]
            max_index = max(indexs)

            task_rates = deque(
                [
                    extract_obj(f"{save_path}/task_{max_index-i}_task_rates.obj")
                    for i in range(max_len)
                ],
                maxlen=max_len,
            )
            task_distortions = deque(
                [
                    extract_obj(f"{save_path}/task_{max_index-i}_task_distortions.obj")
                    for i in range(max_len)
                ],
                maxlen=max_len,
            )
            task_progs = deque(
                [
                    extract_obj(f"{save_path}/task_{max_index-i}_task_progs.obj")
                    for i in range(max_len)
                ],
                maxlen=max_len,
            )
            global_lib_list = deque(
                [
                    extract_obj(f"{save_path}/task_{max_index-i}_global_lib_list.obj")
                    for i in range(max_len)
                ],
                maxlen=max_len,
            )

            start_ind = max_index + 1
            end_ind = start_ind + self.task_num

        else:
            # Fresh start: initialise all deques with dummy zero values
            task_rates = self._construct_deque(self.max_melody_num + 1, [0])
            task_distortions = self._construct_deque(self.max_melody_num + 1, [0])
            task_progs = self._construct_deque(
                self.max_melody_num + 1, [pd.DataFrame()]
            )
            global_lib_list = self._construct_deque(
                self.max_melody_num + 1, [self.init_pm]
            )
            start_ind = self.task_start_ind
            end_ind = self.task_start_ind + self.task_num

        for n in range(start_ind, end_ind):
            logging.info(f"Task {n}/{len(tasks)}")

            if n == 0:
                # First melody: no backtracking possible; run inner DP directly
                rate, distortion, prog_trajs = self.run_inner_dp_per_task(tasks[n])

                # Pad with dummy values for the history window
                task_rates.append([0] * self.max_melody_num + [rate])
                task_distortions.append([0] * self.max_melody_num + [distortion])
                task_progs.append([pd.DataFrame()] * self.max_melody_num + [prog_trajs])

                self.lib.update_post_lib(prog_trajs)
                global_lib_list.append(
                    [self.init_pm] * self.max_melody_num + [self.lib.production]
                )

            else:
                # Subsequent melodies: run outer DP with backtracking
                rate, distortion, prog_trajs, global_lib = self.run_per_task(
                    tasks,
                    n,
                    [task_rates, task_distortions, task_progs, global_lib_list],
                )

                task_rates.append(rate)
                task_distortions.append(distortion)
                task_progs.append(prog_trajs)
                global_lib_list.append(global_lib)

            if save_path:
                save_dict = {
                    "task_rates": task_rates[-1],
                    "task_distortions": task_distortions[-1],
                    "task_progs": task_progs[-1],
                    "global_lib_list": global_lib_list[-1],
                }
                self.save_result_per_task(n, save_dict, save_path)


# ---------------------------------------------------------------------------
# Hierarchical Adaptor Grammar compressor  (global + local libraries)
# ---------------------------------------------------------------------------


class DP_HAGCompressor(DPCompressor):
    """DP compressor with a Hierarchical Adaptor Grammar (HAG).

    Extends :class:`DP_AGCompressor` with a *local* (per-task) library that
    captures patterns reused within a single melody, in addition to the
    *global* library that captures patterns shared across melodies.

    The inner DP (run by :meth:`run_inner_dp_per_task`) maintains its own
    local library deque: as the inner DP segments the melody into sub-melodies
    left to right, the local library grows with each selected sub-program.
    The optimal segmentation therefore depends on the accumulated local
    library.

    The outer DP (run by :meth:`run_per_task`) operates over blocks of past
    melodies exactly as in :class:`DP_AGCompressor`, but additionally tracks
    local library states and the HAG global update table.
    """

    def __init__(self, program_lib: object, args: object) -> None:
        super().__init__(program_lib, args)

        self.max_melody_num = args.max_melody_num
        # For HAG, init_pm refers to the global production table
        self.init_pm = self.lib.global_production
        self.init_table = dict(self.lib.history_table)

    def run_inner_dp_per_task(
        self,
        task: np.ndarray,
        global_lib: Optional[pd.DataFrame] = None,
        global_table: Optional[Dict] = None,
    ) -> Tuple[float, float, pd.DataFrame, pd.DataFrame]:
        """Run the inner DP over one melody with a local library.

        The optimal segmentation also considers the accumulated local library:
        starting from a farther-back position means the local library has
        fewer entries, potentially changing which programs are optimal.

        Args:
            task: Melody array to compress.
            global_lib: Global production table to use (from the outer DP
                cache).  Defaults to ``self.init_pm`` if None.
            global_table: HAG history table to use.  Defaults to
                ``self.init_table`` if None.

        Returns:
            Tuple ``(rate, distortion, prog_list, local_lib_state)``:
            - ``rate``: Total rate for the optimal segmentation.
            - ``distortion``: Total distortion.
            - ``prog_list``: DataFrame of selected programs.
            - ``local_lib_state``: Final local production table after
              processing the full melody (used by the outer DP).
        """
        note_window = self.submelody_backtrack_budget
        task_length = len(task)

        rates = self._construct_deque(note_window + 1, 0)
        distortions = self._construct_deque(note_window + 1, 0)
        prog_list = self._construct_deque(note_window + 1, pd.DataFrame())
        local_lib = self._construct_deque(note_window + 1, self.init_pm)

        # Reset local library to the initial (empty) state
        self.lib.production = self.init_pm

        # Set global library to the state passed in from the outer DP
        self.lib.history_table = (
            global_table if global_table is not None else dict(self.init_table)
        )
        self.lib.global_production = (
            global_lib if global_lib is not None else self.init_pm
        )

        for i in range(1, task_length + 1):
            print(f"Index {i}/{task_length}", flush=True)
            min_cost = float("inf")

            # Baseline: treat the single new note as a fresh sub-melody
            subtask = task[i - 1 : i]
            self.lib.production = local_lib[-1]
            cur_prog = self.run_per_subtask(subtask)
            total_cost = (
                self.beta * (-cur_prog["log_prob"].sum() + rates[-1])
                + cur_prog["distortion"].sum()
                + distortions[-1]
            )
            # Update local library with the baseline choice
            self.lib.update_local_lib(cur_prog)
            min_local_lib = self.lib.production
            min_prog = pd.concat([prog_list[-1], cur_prog], ignore_index=True)
            min_cost = total_cost
            min_rate = -min_prog["log_prob"].sum()
            min_dist = min_prog["distortion"].sum()

            # Inner DP: try extending back up to note_window positions
            for l in range(1, min(note_window, i) + 1):
                subtask = task[i - l : i]
                # Restore local library to the state that was active l steps ago
                self.lib.production = local_lib[-l]

                cur_prog = self.run_per_subtask(subtask)
                cur_rate = -cur_prog["log_prob"].sum()
                cur_dist = cur_prog["distortion"].sum()

                # D + β·R (cumulative)
                total_cost = (
                    self.beta * (cur_rate + rates[-l]) + cur_dist + distortions[-l]
                )

                if total_cost < min_cost:
                    min_cost = total_cost
                    min_rate = cur_rate + rates[-l]
                    min_dist = cur_dist + distortions[-l]
                    min_prog = pd.concat([prog_list[-l], cur_prog], ignore_index=True)

                    self.lib.update_local_lib(cur_prog)
                    min_local_lib = self.lib.production

            rates.append(min_rate)
            distortions.append(min_dist)
            prog_list.append(min_prog)
            local_lib.append(min_local_lib)

        return rates[-1], distortions[-1], prog_list[-1], local_lib[-1]

    def run_per_task(
        self,
        tasks: List[np.ndarray],
        cur_task_ind: int,
        cache: List,
    ) -> Tuple[List, List, List, List, List, List]:
        """Outer DP step for HAG: find the optimal melody-level backtracking.

        Args:
            tasks: Full list of melody arrays.
            cur_task_ind: Index of the current melody.
            cache: List of six deques:
                ``[rates, distortions, progs, global_libs, global_tables, local_libs]``.

        Returns:
            Six lists (sliced to the last ``max_melody_num+1`` entries) for
            the optimal block:
            ``(rates, distortions, prog_trajs, global_libs, global_tables, local_libs)``.
        """
        max_melody_num = self.max_melody_num
        (
            cache_rates,
            cache_distortions,
            cache_progs,
            cache_global_lib,
            cache_global_table,
            cache_local_lib,
        ) = cache

        min_cost = float("inf")

        for l in range(0, min(max_melody_num, cur_task_ind) + 1):
            temp_ind = -(l + 1)
            # Retrieve cached history from l melodies ago
            temp_progs = list(cache_progs[temp_ind])
            temp_rates = list(cache_rates[temp_ind])
            temp_distortions = list(cache_distortions[temp_ind])
            temp_local_libs = list(cache_local_lib[temp_ind])
            temp_global_libs = list(cache_global_lib[temp_ind])
            temp_global_tables = list(cache_global_table[temp_ind])

            # Re-process melodies [n-l, …, n] from the recovered library state
            for n in range(cur_task_ind - l, cur_task_ind + 1):
                task = tasks[n]
                self.lib.production = self.init_pm

                if n == 0:
                    # First melody: use the initial (empty) library
                    self.lib.history_table = self.init_table
                    self.lib.global_production = self.init_pm
                    (
                        r_per_task,
                        d_per_task,
                        prog_per_task,
                        local_lib_per_task,
                    ) = self.run_inner_dp_per_task(task)
                else:
                    # Use the global library / table state from the cache
                    (
                        r_per_task,
                        d_per_task,
                        prog_per_task,
                        local_lib_per_task,
                    ) = self.run_inner_dp_per_task(
                        task, temp_global_libs[-1], dict(temp_global_tables[-1])
                    )

                temp_progs.append(prog_per_task)
                temp_rates.append(r_per_task)
                temp_distortions.append(d_per_task)
                temp_local_libs.append(local_lib_per_task)

                # Update global library after processing this melody
                self.lib.update_post_lib(prog_per_task, local_lib_per_task)
                temp_global_libs.append(self.lib.global_production)
                temp_global_tables.append(dict(self.lib.history_table))

            cost = sum(temp_distortions) + self.beta * sum(temp_rates)

            if cost < min_cost:
                min_cost = cost
                min_rate = temp_rates
                min_distortion = temp_distortions
                min_prog_traj = temp_progs
                min_local_lib = temp_local_libs
                min_global_lib = temp_global_libs
                min_global_table = temp_global_tables

        slice_start = -max_melody_num - 1
        return (
            min_rate[slice_start:],
            min_distortion[slice_start:],
            min_prog_traj[slice_start:],
            min_global_lib[slice_start:],
            min_global_table[slice_start:],
            min_local_lib[slice_start:],
        )

    def run(self, tasks: List[np.ndarray], save_path: str) -> None:
        """Compress all tasks with HAG global + local library updates.

        Maintains six rolling deques (rates, distortions, programs, local
        library states, global library states, global HAG tables) of length
        ``max_melody_num + 1``.  Supports resuming via ``--continue_infer``.

        Args:
            tasks: List of melody arrays.
            save_path: Output directory for per-task pickle files.
        """
        max_len = self.max_melody_num + 1

        if self.continue_infer:
            # Resume: reload the last max_len saved checkpoints
            prog_names = [
                file_name for file_name in os.listdir(save_path) if "prog" in file_name
            ]
            indexs = [
                int(re.search(r"task_(\d+)_", file_name).group(1))
                for file_name in prog_names
            ]
            max_index = max(indexs)

            task_rates = deque(
                [
                    extract_obj(f"{save_path}/task_{max_index-i}_task_rates.obj")
                    for i in range(max_len)
                ],
                maxlen=max_len,
            )
            task_distortions = deque(
                [
                    extract_obj(f"{save_path}/task_{max_index-i}_task_distortions.obj")
                    for i in range(max_len)
                ],
                maxlen=max_len,
            )
            task_progs = deque(
                [
                    extract_obj(f"{save_path}/task_{max_index-i}_task_progs.obj")
                    for i in range(max_len)
                ],
                maxlen=max_len,
            )
            local_lib_list = deque(
                [
                    extract_obj(f"{save_path}/task_{max_index-i}_local_lib_list.obj")
                    for i in range(max_len)
                ],
                maxlen=max_len,
            )
            global_lib_list = deque(
                [
                    extract_obj(f"{save_path}/task_{max_index-i}_global_lib_list.obj")
                    for i in range(max_len)
                ],
                maxlen=max_len,
            )
            global_table_list = deque(
                [
                    extract_obj(f"{save_path}/task_{max_index-i}_global_table_list.obj")
                    for i in range(max_len)
                ],
                maxlen=max_len,
            )
            start_ind = max_index + 1
            end_ind = self.task_num + start_ind

        else:
            # Fresh start: initialise all deques with dummy zero / empty values
            task_rates = self._construct_deque(max_len, [0])
            task_distortions = self._construct_deque(max_len, [0])
            task_progs = self._construct_deque(max_len, [pd.DataFrame()])
            local_lib_list = self._construct_deque(max_len, [self.init_pm])
            global_lib_list = self._construct_deque(max_len, [self.init_pm])
            global_table_list = self._construct_deque(max_len, [dict(self.init_table)])

            start_ind = self.task_start_ind
            end_ind = self.task_start_ind + self.task_num

        for n in range(start_ind, end_ind):
            logging.info(f"Task {n}/{len(tasks)}")

            if n == 0:
                # First melody: no backtracking; run inner DP directly
                (
                    r_per_task,
                    d_per_task,
                    prog_per_task,
                    local_lib_per_task,
                ) = self.run_inner_dp_per_task(tasks[n])

                # Pad history window with dummy values
                task_rates.append([0] * self.max_melody_num + [r_per_task])
                task_distortions.append([0] * self.max_melody_num + [d_per_task])
                task_progs.append(
                    [pd.DataFrame()] * self.max_melody_num + [prog_per_task]
                )
                local_lib_list.append(
                    [self.init_pm] * self.max_melody_num + [local_lib_per_task]
                )

                # Update global library and record history table snapshot
                prim_table = self.lib.history_table.copy()
                self.lib.update_post_lib(prog_per_task, local_lib_per_task)
                global_lib_list.append(
                    [self.init_pm] * self.max_melody_num + [self.lib.global_production]
                )
                global_table_list.append(
                    [prim_table] * self.max_melody_num + [self.lib.history_table]
                )

            else:
                # Subsequent melodies: run outer DP with backtracking
                (
                    rate,
                    distortion,
                    prog_trajs,
                    global_lib,
                    global_table,
                    local_lib,
                ) = self.run_per_task(
                    tasks,
                    n,
                    [
                        task_rates,
                        task_distortions,
                        task_progs,
                        global_lib_list,
                        global_table_list,
                        local_lib_list,
                    ],
                )

                task_rates.append(rate)
                task_distortions.append(distortion)
                task_progs.append(prog_trajs)

                local_lib_list.append(local_lib)
                global_lib_list.append(global_lib)
                global_table_list.append(global_table)

            if save_path:
                save_dict = {
                    "task_rates": task_rates[-1],
                    "task_distortions": task_distortions[-1],
                    "task_progs": task_progs[-1],
                    "local_lib_list": local_lib_list[-1],
                    "global_lib_list": global_lib_list[-1],
                    "global_table_list": global_table_list[-1],
                }
                self.save_result_per_task(n, save_dict, save_path)
