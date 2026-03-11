"""
greedy_dp_compressor.py — Greedy DP compressors for melody program induction.

Implements a family of dynamic-programming compressors where program selection
at each sub-melody step is *stochastic* (sampled proportional to exp(value))
rather than exhaustive as in ``dp_compressor.py``.  This makes inference
faster at the cost of guaranteed optimality.

Class hierarchy
---------------
::

    Compressor  (base_compressor.py)
    └── GreedyDPCompressor
        └── GreedyDP_PCFGCompressor   — fixed PCFG prior, no library update
            └── GreedyDP_AGCompressor — global library updated after each task
                └── GreedyDP_HAGCompressor — global + per-task local library

Key algorithmic differences from the normative compressor
----------------------------------------------------------
* **Program selection**: normative uses argmax; greedy uses weighted sampling
  ``progs.sample(n=1, weights=exp(value))`` — stochastic but faster.
* **Inner DP** (within a melody): both perform 1-D DP over notes with a
  backtracking window of ``submelody_backtrack_budget`` notes.
* **Outer DP** (across melodies): AG/HAG variants backtrack over
  ``melody_backtrack_budget`` previous melodies to find the library state that
  minimises the cumulative RD cost.

Human-experiment helpers
------------------------
``run_submelody_human`` and ``run_compose_human`` methods are used for
analysing model predictions against human behavioural data (CHIRP dataset).
They are not part of the standard training pipeline.
"""

import re
import math
import numpy as np
import pandas as pd
from collections import deque
from typing import List, Optional, Tuple

pd.options.mode.chained_assignment = None

from src.program.router import *
from src.domain.melody.melody_primitive import (
    MelodyProgram,
    Note,
    create_or_get_pm_from_cache,
)
from src.domain.melody.base_compressor import Compressor

EPS = 1e-6
NUM_MELODY = 5
NUM_SUBMELODY = 6


class GreedyDPCompressor(Compressor):
    """Base class for greedy DP melody compressors.

    Extends :class:`Compressor` with frame generation, stochastic program
    selection, and a greedy inner-DP loop over sub-melodies.  Not intended
    to be instantiated directly — use one of the three curriculum subclasses.
    """

    def __init__(self, program_lib: object, args: object) -> None:
        super().__init__(program_lib, args)

        # Experiment setting and hyperparameters
        self.submelody_backtrack_budget = args.submelody_backtrack_budget
        self.lossless = args.lossless
        self.frame_gen = args.frame_gen

        # Save the initial production memory
        self.init_pm = self.lib.production

        # Memory & computation constraint
        self.reuse_num_note_subtask = args.reuse_num_note_subtask

        self.curriculum = args.curriculum

        # Frame library
        self.sample_subprog = "switch"  # uniform
        self.switch = math.exp(-1) / (math.exp(-1) + math.exp(-2))

        # Args
        self.args = args

    @staticmethod
    @np.vectorize
    def _reconstruct_subtask(cur_prog_term, len_sub_task):
        """
        Reconstruct the subtask based on the current program
        """
        cur_prog = MelodyProgram(cur_prog_term)

        result = cur_prog.run()[:len_sub_task]

        return result

    @staticmethod
    def _comp_subprog_recon_len(
        cur_all_progs: pd.DataFrame,
        subtask: np.ndarray,
    ) -> pd.DataFrame:
        """Evaluate each program and record its reconstruction length.

        Runs every program term in ``cur_all_progs`` and truncates the output
        to ``len(subtask)`` notes, then stores the result and its length.

        Args:
            cur_all_progs: DataFrame with at least a ``"term"`` column
                containing program strings.
            subtask: Ground-truth note sequence for the current sub-melody.

        Returns:
            The same DataFrame with two new columns:
            ``"recon"`` (truncated reconstruction array) and
            ``"recon_len"`` (length of that reconstruction).
        """
        # Evaluate the program
        cur_all_progs["recon"] = cur_all_progs["term"].apply(
            GreedyDPCompressor._reconstruct_subtask, args=(len(subtask),)
        )

        # Compute the reconstruction length
        cur_all_progs["recon_len"] = cur_all_progs["recon"].apply(len)
        return cur_all_progs

    def _calculate_cost(self, program: pd.DataFrame) -> float:
        """Compute the RD cost of a program sequence.

        Args:
            program: DataFrame of sub-programs, each row representing one
                segment of the melody.  Must contain ``"distortion"`` and
                ``"log_prob"`` columns.

        Returns:
            Scalar RD cost (lower is better).
        """
        return program["distortion"].sum() - self.beta * program["log_prob"].sum()

    def _fill_single_frame(self, sampled_frame: pd.Series) -> dict:
        """Helper function to process a single frame and return the filled program."""
        # Process a single frame using unfold_frame
        filled_prog = self.lib.unfold_frame(
            sampled_frame["term"], sampled_frame["type_string"]
        )
        # Add additional fields
        filled_prog["frame"] = sampled_frame[
            "frame"
        ]  # TODO: have to make sure no hold in the frame
        filled_prog["depth"] = sampled_frame["depth"]

        return filled_prog

    def _initialize_deques(self, note_window):
        """Initialize deques for rates, distortions, programs, and start indices."""
        return (
            self._construct_deque(note_window + 1, 0),  # rates
            self._construct_deque(note_window + 1, 0),  # distortions
            self._construct_deque(note_window + 1, pd.DataFrame()),  # prog_list
            self._construct_deque(note_window + 1, float(-np.inf)),  # start_inds
        )

    def add_frame_for_mem_constrained_by_len(self, subtask: np.ndarray) -> pd.DataFrame:
        """Build a memorise program that exactly covers the full subtask.

        Args:
            subtask: Ground-truth note sequence to be memorised.

        Returns:
            A single-row DataFrame with the memorise program, its log-prior,
            distortion (0), reconstruction, and value.
        """
        num_note = len(subtask)

        note_name = Note.array_to_string(subtask)
        possible_notes = create_or_get_pm_from_cache(note_name)

        log_prob_memorize = np.log(0.25) + np.log(1 / 7)
        log_prob_note = np.log(1 / 6)

        log_prog_whole = log_prob_memorize + log_prob_note * num_note
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

        memorized_progs = self.comp_ll_value(memorized_progs)
        return memorized_progs

    def run_per_subtask(
        self,
        subtask: np.ndarray,
        frames: Optional[pd.DataFrame] = None,
        mask_data: Optional[np.ndarray] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Sample a program for a single sub-melody segment.

        Generates candidate programs for ``subtask``, optionally adds a
        memorise program, and *stochastically* selects one program
        proportional to exp(value).  This is the core greedy step — unlike
        the normative compressor, which takes the argmax.

        Args:
            subtask: Note sequence for the current sub-melody chunk.
            frames: Pre-sampled frame DataFrame.  If ``None``, frames are
                generated on the fly via :meth:`fill_frame`.
            mask_data: Per-note correctness mask for human-experiment
                analysis.  If not ``None``, used to filter programs in
                ``lossless_error`` mode.

        Returns:
            A ``(frames, chosen_subprog)`` tuple where ``frames`` is the
            (possibly freshly generated) frame DataFrame and
            ``chosen_subprog`` is a single-row DataFrame representing the
            selected program.
        """

        def add_memorization_program() -> pd.DataFrame:
            """Add a cheating program for memorization if the 'mem' argument is enabled."""
            return (
                self.add_frame_for_memorization(subtask)
                if self.args.mem
                else pd.DataFrame()
            )

        def sample_or_unfold_frames():
            """Handle frame sampling or unfolding based on the availability of frames."""
            if frames is None:
                return sample_frames()
            return unfold_frames()

        def sample_frames():
            """Sample frames and apply filtering if lossless error is enabled."""
            # It sets to reuse the generated/sampled frames when the inference goes back so that we can keep the computational cost
            progs, frames = self.fill_frame(subtask)

            if self.args.lossless_error:
                progs = filter_lossless_error(progs, mask_data)

            return progs, frames

        def filter_lossless_error(progs: pd.DataFrame, mask_data) -> pd.DataFrame:
            """Filter programs based on the lossless error condition."""
            human_correct_msk = mask_data == subtask
            save_mask = [
                i
                for i in range(len(progs))
                if (
                    human_correct_msk[: progs["recon_len"][i]].sum()
                    == (
                        human_correct_msk[: progs["recon_len"][i]]
                        & (progs["recon"][i] == subtask[: progs["recon_len"][i]])
                    ).sum()
                )
            ]
            return progs.loc[save_mask].reset_index(drop=True)

        def unfold_frames():
            """Unfold pre-sampled frames into programs."""
            unfolded_progs = []
            for _, frame in frames.iterrows():
                unfolded_progs.append(process_single_frame(frame))

            return pd.concat(unfolded_progs, ignore_index=True), frames

        def process_single_frame(frame: pd.Series) -> pd.DataFrame:
            """Process a single frame to generate a program."""
            frame_depth, frame_term, type_string = (
                frame["depth"],
                frame["frame"],
                frame["type_string"],
            )

            prog = self.lib.unfold_frame(frame_term, type_string)
            if prog.empty:
                prog = self.lib.unfold_frame(
                    frame_term, type_string, production=self.lib.global_production
                )

            prog = self._comp_subprog_recon_len(prog, subtask)
            prog = self._comp_subprog_value(prog, subtask)

            if not self.args.lossless_error:
                prog = prog.loc[[prog["value"].idxmax()]]

            prog["depth"], prog["frame"] = frame_depth, frame_term
            return prog

        # Add cheating program for memorization if needed
        mem_prog = add_memorization_program()

        # Sample or unfold frames
        progs, frames = sample_or_unfold_frames()

        # For DP, we do not distinguish frames;
        # Instead, we will constrain the length to be the same as subtask
        progs = pd.concat([progs, mem_prog], ignore_index=True)

        # If self.lossless_error is enabled, we will filter the programs
        if self.lossless:
            progs = progs[progs["log_ll"] > 0]

        # Sample one subprograms among all candidates
        # NOTE: this is different than the original dp which chooses the best program
        chosen_subprog = (
            progs.sample(n=1, weights=np.exp(progs["value"]) + EPS)
            if len(progs) > 1
            else progs
        )
        return frames, chosen_subprog

    def _fill_frame_fly(
        self, subtask: np.ndarray
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Generate and fill program frames on the fly for one sub-melody.

        Samples ``search_budget`` (type string, depth) pairs, generates a program
        frame for each using :meth:`lib.generate_frame`, evaluates all filled
        instantiations against ``subtask``, and keeps the best-valued one per
        frame.

        Args:
            subtask: Ground-truth note sequence for the current sub-melody.

        Returns:
            A ``(filled_progs, unfilled_frames)`` tuple.  ``filled_progs`` is
            a DataFrame of the best-valued program per frame (one row each);
            ``unfilled_frames`` records the raw frame metadata.  Returns
            ``(empty DataFrame, None)`` if all sampled frames were empty.
        """
        filled_progs = []
        unfilled_frames = []

        # Sample the type string and depth
        def _sample_search_budget_power_law(search_budget, alpha=2.0):
            if search_budget == 1:
                return 1
            rng = np.random.default_rng()
            values = np.arange(1, search_budget + 1)
            weights = values ** (-alpha)
            probs = weights / weights.sum()
            return int(rng.choice(values[::-1], p=probs))

        # search_budget = _sample_search_budget_power_law(self.search_budget)
        search_budget = int(self.search_budget)
        type_string_list = self.sample_type_string_subtask(num_sample=search_budget)
        depth_list = self.sample_depth_subtask(num_sample=search_budget)

        for type_string, depth in zip(type_string_list, depth_list):
            # Generate the frames and fill them with arguments
            filled_prog = self.lib.generate_frame(type_string, depth)
            # NOTE: add this condition to avoid empty frame;
            # Which only happens when the local library size is small, it will filter out some programs
            # But later we will refill the program given frame, when there is unfilled part (PM) in the frame where certain type strings have been filtered out
            # This will result in unsuccesful filling, but we will keep the frame
            if filled_prog.empty:
                continue
            # Add the frame information
            unfilled_frames.append(
                filled_prog.iloc[[0]][["frame", "type_string", "depth", "log_prob"]]
            )
            # Compute the reconstruction length and value
            filled_prog = self._comp_subprog_recon_len(filled_prog, subtask)
            filled_prog = self._comp_subprog_value(filled_prog, subtask)

            # Choose one filled programs among all candidates
            filled_prog = filled_prog.loc[[filled_prog["value"].idxmax()]]
            filled_progs.append(filled_prog)
            # TODO: we could also sample one to add some randomness
            # sampled_frame.sample(
            #         n=1, weights=np.exp(sampled_frame["value"]) + EPS
            #     )

        # Return the filled programs and unfilled frames
        if len(filled_progs) == 0:
            return pd.DataFrame(), None
        else:
            return pd.concat(filled_progs, ignore_index=True), pd.concat(
                unfilled_frames, ignore_index=True
            )

    def fill_frame(
        self,
        subtask: np.ndarray,
        index: list = [0, 0],
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Dispatch frame generation based on the configured strategy.

        Args:
            subtask: Ground-truth note sequence for the current sub-melody.
            index: ``[task_index, subtask_index]`` — currently unused but
                reserved for pre-sampled per-task frame files.

        Returns:
            A ``(filled_progs, frames)`` tuple as returned by the underlying
            frame-generation method.
        """
        if self.args.frame_gen == "fly":
            return self._fill_frame_fly(subtask)

        elif self.frame_gen == "fly_lossless":  # TODO
            filled_progs = []
            unfilled_frames = []
            for _ in range(self.search_budget):
                filled_prog = pd.empty()
                while filled_prog.empty:
                    type_string = self.sample_type_string_subtask(num_sample=1)[0]
                    depth = self.sample_depth_subtask(num_sample=1)[0]
                    filled_prog = self.lib.generate_frame(type_string, depth)

                # Add the frame information
                unfilled_frames.append(
                    filled_prog.iloc[[0]][["frame", "type_string", "depth", "log_prob"]]
                )
                # Sample one filled programs among all candidates
                filled_prog = self.comp_subprog_recon_len(filled_prog, subtask)
                filled_prog = self.comp_subprog_value(filled_prog, subtask)
                filled_progs.append(filled_prog)
            return pd.concat(filled_progs, ignore_index=True), pd.concat(
                unfilled_frames, ignore_index=True
            )
        elif self.frame_gen == "query_all":  # TODO: for temporary use
            cur_all_progs = self.args.frames
            cur_all_progs["distortion"] = cur_all_progs["recon"].apply(
                self._comp_subprog_distortion, args=(subtask,)
            )
            cur_all_progs = self.comp_ll_value(cur_all_progs)
        else:
            raise NotImplementedError


class GreedyDP_PCFGCompressor(GreedyDPCompressor):
    """Greedy DP compressor with a fixed PCFG prior (no library updates).

    Uses a fixed probabilistic context-free grammar throughout all tasks.
    The grammar is never updated, so all programs are evaluated against the
    same static prior.  This is the simplest curriculum and serves as a
    baseline for the adaptive grammar variants.

    The inner DP (``run_per_task``) greedily segments a single melody into
    sub-programs, backtracking up to ``submelody_backtrack_budget`` notes to find
    a cheaper segmentation.
    """

    def __init__(self, program_lib: object, args: object) -> None:
        super().__init__(program_lib, args)

    def _run_forward_simulation_one_step(
        self,
        task: np.ndarray,
        frames: Optional[pd.DataFrame] = None,
        mask_data: Optional[np.ndarray] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run one greedy step: sample a single program for the current subtask.

        Args:
            task: Remaining (uncompressed) note sequence starting at the
                current DP position.
            frames: Pre-sampled frames to reuse (or ``None`` to generate anew).
            mask_data: Per-note human correctness mask (for human analysis).

        Returns:
            A ``(frames, chosen_subprog)`` tuple from :meth:`run_per_subtask`.
        """
        return self.run_per_subtask(task, frames, mask_data)

    def _run_forward_simulation_multi_step(self, task: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate the forward process for multiple steps over a given task.

        Parameters:
            task (pd.DataFrame): The input task sequence to process.

        Returns:
            pd.DataFrame: A concatenated DataFrame containing all filled subtask programs.
        """
        # Initialize parameters
        start_ind = 0
        frames = None
        filled_prog = []

        while start_ind < len(task):
            # Run simulation for the current subtask slice
            frames, filled = self.run_per_subtask(task[start_ind:], frames)

            # Append the filled program and update the start index
            filled_prog.append(filled)
            start_ind += filled["recon_len"].values[0]

        return pd.concat(filled_prog, ignore_index=True)

    @staticmethod
    def _pad_with_inf(lst: list, target_length: int) -> list:
        """Prepend ``-inf`` sentinels to a list until it reaches ``target_length``.

        Used to align cached cost/program lists to a fixed window size so that
        deque indexing is consistent regardless of how many tasks have been
        processed so far.

        Args:
            lst: List to pad.
            target_length: Desired total length.

        Returns:
            The original list if already long enough, otherwise a new list
            with ``-inf`` prepended to reach ``target_length``.
        """
        if len(lst) >= target_length:
            return lst  # No padding needed
        # Calculate the number of `np.inf` to add
        padding = [float(-np.inf)] * (target_length - len(lst))
        # Prepend the padding to the list
        return padding + lst

    def run_per_task(self, task, mask_data=None) -> tuple:
        """
        Process a task sequence by iteratively calculating compression rates and distortions.

        Parameters:
            task (pd.DataFrame): The sequence to process.
            task_ind (int): Index of the task being processed (default is 0).

        Returns:
            tuple: Contains the final rate, distortion, and the program list.
        """

        def _sample_backtrack_length(submelody_backtrack_budget, alpha=2.0):
            """
            Sample the backtrack length (integer) from the powerlaw distribution with the maximum length as submelody_backtrack_budget.
            E.g., if submelody_backtrack_budget is 4, the backtrack length can be 1, 2, 3, or 4.
            The probability of each length is according to the powerlaw distribution.
            """
            if submelody_backtrack_budget == 0:
                return 0
            rng = np.random.default_rng()
            values = np.arange(1, submelody_backtrack_budget + 2)
            # power-law weights ~ 1/k^alpha
            weights = values ** (-alpha)
            probs = weights / weights.sum()

            return int(rng.choice(values[::-1], p=probs) - 1)

        # Task information
        task_len = len(task)
        mask_data = mask_data if mask_data is not None else task

        # Initialize parameters
        # note_window = _sample_backtrack_length(self.submelody_backtrack_budget)
        note_window = int(self.submelody_backtrack_budget)
        rates, distortions, prog_list, start_inds = self._initialize_deques(note_window)

        # Iterate from the start of the sequence to the end
        i = 0
        while i < len(task):
            # print(f"Index {i}/{task_len}", flush=True)

            # Initialize the start index in case no better submelody is found
            start_inds.append(i)
            _, first_filled = self._run_forward_simulation_one_step(
                task[i:], mask_data=mask_data[i:]
            )

            # Concatenate new program with the previous one to get new rate and distortion
            min_prog = pd.concat([prog_list[-1], first_filled], ignore_index=True)
            cur_recon_len = min(first_filled["recon_len"].values[0], task_len - i)
            i += cur_recon_len
            min_cost = self._calculate_cost(min_prog)

            # Look back within the note window to find a better submelody
            for l in range(1, note_window + 1):
                start_ind = start_inds[-l]

                if start_ind < 0:
                    break

                # Calculate reward for compressing the submelody sequence[i-l:i]
                new_prog = self._run_forward_simulation_multi_step(task[start_ind:i])

                # Combine the new program with the previous one
                new_prog = pd.concat([prog_list[-l], new_prog], ignore_index=True)
                new_cost = self._calculate_cost(new_prog)

                # Update the minimum cost and program if a better submelody is found
                if new_cost < min_cost:
                    min_cost = new_cost
                    min_prog = new_prog

            rates.append(-min_prog["log_prob"].sum())
            distortions.append(min_prog["distortion"].sum())
            prog_list.append(min_prog)

        return rates[-1], distortions[-1], prog_list[-1]

    def run_submelody_human(
        self,
        tasks: List[pd.DataFrame],
        part_id: int,
        save_path: str,
        mask_data: Optional[list] = None,
    ) -> Tuple[list, dict]:
        """Run the PCFG compressor on human-experiment sub-melody data.

        Processes ``NUM_MELODY`` melodies × ``NUM_SUBMELODY - 1`` sub-melodies
        per participant and saves per-participant results.  Used for fitting
        the model to the CHIRP behavioural dataset.

        Args:
            tasks: Nested list of shape ``[NUM_MELODY][NUM_SUBMELODY]``
                containing note arrays for each sub-melody.
            part_id: Participant identifier (used to name the output file).
            save_path: Directory to write results to.
            mask_data: Optional correctness mask with the same shape as
                ``tasks``, used in ``lossless_error`` mode.

        Returns:
            A ``(progs, results)`` tuple where ``progs`` is a nested list of
            program DataFrames and ``results`` is a dict with keys
            ``"recon"``, ``"rate"``, ``"distortion"``.
        """
        progs = []
        results = {"recon": [], "rate": [], "distortion": []}

        for i in range(NUM_MELODY):
            progs_per_task, recon_per_task, rate_per_task, distortion_per_task = (
                [],
                [],
                [],
                [],
            )
            for j in range(NUM_SUBMELODY - 1):
                cur_task = tasks[i][j]

                _, _, prog_trajs = self.run_per_task(
                    cur_task,
                    mask_data=mask_data[i][j] if mask_data is not None else None,
                )
                progs_per_task.append(prog_trajs)

                recon = []
                for prog_ind in range(len(prog_trajs)):
                    recon += prog_trajs["recon"][prog_ind].tolist()
                recon_per_task.append(recon)
                rate_per_task.append(-prog_trajs["log_prob"].sum())
                distortion_per_task.append(prog_trajs["distortion"].sum())

            progs.append(progs_per_task)
            results["recon"].append(recon_per_task)
            results["rate"].append(rate_per_task)
            results["distortion"].append(distortion_per_task)

        self.save_result_per_task(f"p{part_id}", {"prog_trajs": progs}, save_path)
        self.save_result_per_task(f"p{part_id}", {"results": results}, save_path)

        return progs, results

    def run_compose_human(
        self,
        task: np.ndarray,
    ) -> Tuple[pd.DataFrame, list, float, float]:
        """Run the PCFG compressor on a single melody (composition task).

        Convenience wrapper for human-experiment analysis that returns the
        program trajectory and summary statistics for one melody.

        Args:
            task: Note sequence to compress.

        Returns:
            A ``(prog_trajs, recon, rate, distortion)`` tuple where
            ``prog_trajs`` is a DataFrame of selected sub-programs,
            ``recon`` is the concatenated reconstruction as a list,
            ``rate`` is the total description length (−log_prob), and
            ``distortion`` is the total Levenshtein distortion.
        """
        _, _, prog_trajs = self.run_per_task(task, None)
        # Update the results
        recon = []
        for prog_ind in range(len(prog_trajs)):
            recon += prog_trajs["recon"][prog_ind].tolist()

        return (
            prog_trajs,
            recon,
            -prog_trajs["log_prob"].sum(),
            prog_trajs["distortion"].sum(),
        )

    def run(
        self,
        tasks: List[pd.DataFrame],
        save_path: str,
    ) -> None:
        """
        This is the main function for sampling programs for all tasks
        Args:
            tasks: a list of dataframes, each of which contains the task information
            save: whether to save the sampled programs
        """
        # ----- Outer loop: iteration over all tasks
        for n in range(self.task_start_ind, self.task_start_ind + self.task_num):
            # Task information
            cur_task = tasks[n]
            print(f"Task {n}/{len(tasks)}", flush=True)

            # Find programs that are consistent with data, and compute the likelihood of each program
            _, _, prog_trajs = self.run_per_task(cur_task)

            # Save the results
            if save_path:
                save_dict = {"prog_trajs": prog_trajs}
                self.save_result_per_task(n, save_dict, save_path)

        return prog_trajs


class GreedyDP_AGCompressor(GreedyDP_PCFGCompressor):
    """Greedy DP compressor with a global Adaptor Grammar library.

    Extends :class:`GreedyDP_PCFGCompressor` with an outer DP loop that
    backtracks over up to ``melody_backtrack_budget`` previous melodies.  After
    choosing the optimal melody window, the global Adaptor Grammar library
    is updated with the selected programs via ``lib.update_post_lib``.
    """

    def __init__(self, program_lib: object, args: object) -> None:
        super().__init__(program_lib, args)
        self.melody_backtrack_budget = args.melody_backtrack_budget

    def run_inner_dp_per_task(
        self,
        task: np.ndarray,
        mask_data: Optional[np.ndarray] = None,
    ) -> tuple:
        """Run the inner greedy DP for a single melody (no library update).

        Args:
            task: Note sequence to compress.
            mask_data: Optional per-note correctness mask (human analysis).

        Returns:
            A ``(rate, distortion, prog_trajs)`` tuple from the inner DP.
        """
        # import ipdb; ipdb.set_trace()
        return GreedyDP_PCFGCompressor.run_per_task(self, task, mask_data)

    def run_per_task(
        self, tasks: pd.DataFrame, cur_task_ind: int, cache: list
    ) -> tuple:
        """
        Process a sequence of melodies using dynamic programming, minimizing a cost function.

        Args:
            tasks (pd.DataFrame): A DataFrame containing task information.
            cur_task_ind (int): The current task index to process.
            cache (list): A list containing cached values for rates, distortions, programs, and libraries.

        Returns:
            tuple: A tuple containing:
                - min_rate (list): Rates of the optimal submelodies.
                - min_distortion (list): Distortions of the optimal submelodies.
                - min_prog_traj (list): Programs of the optimal submelodies.
                - min_global_lib (list): Updated global library states.
        """
        melody_backtrack_budget = self.melody_backtrack_budget
        cache_rates, cache_distortions, cache_progs, cache_global_lib = cache

        # Initialize minimum cost and results
        min_cost = float("inf")
        min_rate, min_distortion, min_prog_traj, min_global_lib = [], [], [], []

        # Iterate over possible submelody lengths ending at the current index
        for l in range(0, min(melody_backtrack_budget, cur_task_ind) + 1):
            temp_index = -(l + 1)

            # Extract cached values
            temp_progs = list(cache_progs[temp_index])
            temp_libs = list(cache_global_lib[temp_index])
            temp_rates = list(cache_rates[temp_index])
            temp_distortions = list(cache_distortions[temp_index])

            # Set the library to the last cached production state
            self.lib.production = temp_libs[-1]

            # Process tasks in the submelody
            for n in range(cur_task_ind - l, cur_task_ind + 1):
                task = tasks[n]

                # Compute reward metrics for the current task
                cur_rate, cur_dist, cur_prog = self.run_inner_dp_per_task(task)

                # Update temporary lists
                temp_progs.append(cur_prog)
                temp_rates.append(cur_rate)
                temp_distortions.append(cur_dist)

                # Update the library
                self.lib.update_post_lib(cur_prog)
                temp_libs.append(self.lib.production)

            # Compute the cost for the current submelody
            cost = sum(temp_distortions) + self.beta * sum(temp_rates)

            # Update the minimum cost and results if a better solution is found
            if cost < min_cost:
                min_cost = cost
                min_rate = temp_rates
                min_distortion = temp_distortions
                min_prog_traj = temp_progs
                min_global_lib = temp_libs

        # Return the optimal submelody information, sliced to the maximum melody number
        slice_start = -melody_backtrack_budget - 1
        return (
            min_rate[slice_start:],
            min_distortion[slice_start:],
            min_prog_traj[slice_start:],
            min_global_lib[slice_start:],
        )

    def run_submelody_human(
        self,
        tasks: List[pd.DataFrame],
        part_id: int,
        save_path: str,
        mask_data=None,
    ) -> None:
        """
        This is the main function for sampling programs for all tasks
        Args:
            tasks: a list of dataframes, each of which contains the task information
            save: whether to save the sampled programs
        """
        progs = []
        global_libs = []
        results = {"recon": [], "rate": [], "distortion": []}

        self.lib.production = self.init_pm

        for i in range(NUM_MELODY):
            progs_per_task, recon_per_task, rate_per_task, distortion_per_task = (
                [],
                [],
                [],
                [],
            )

            for j in range(NUM_SUBMELODY - 1):
                cur_task = tasks[i][j]
                _, _, prog_trajs = self.run_inner_dp_per_task(
                    cur_task, mask_data[i][j] if mask_data is not None else None
                )
                progs_per_task.append(prog_trajs)

                recon = []
                for prog_ind in range(len(prog_trajs)):
                    recon += prog_trajs["recon"][prog_ind].tolist()
                recon_per_task.append(recon)
                rate_per_task.append(-prog_trajs["log_prob"].sum())
                distortion_per_task.append(prog_trajs["distortion"].sum())

            # Not update the libaray using the last submelody
            for j in range(NUM_SUBMELODY - 1):
                cur_prog_list = progs_per_task[j]
                for prog_ind in range(len(cur_prog_list)):
                    one_prog = cur_prog_list.iloc[[prog_ind]]
                    self.lib.update_post_lib(one_prog)

            global_libs.append(self.lib.production)
            progs.append(progs_per_task)
            results["recon"].append(recon_per_task)
            results["rate"].append(rate_per_task)
            results["distortion"].append(distortion_per_task)

        self.save_result_per_task(f"p{part_id}", {"prog_trajs": progs}, save_path)
        self.save_result_per_task(f"p{part_id}", {"results": results}, save_path)
        self.save_result_per_task(
            f"p{part_id}", {"global_libs": global_libs}, save_path
        )

        return progs, results

    def run_compose_human(
        self,
        task: np.ndarray,
        global_lib: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, list, float, float]:
        """Run the AG compressor on a single melody given a fixed global library.

        Loads the provided ``global_lib`` into the grammar before running the
        inner DP, so predictions are conditioned on a participant's learned
        library state.

        Args:
            task: Note sequence to compress.
            global_lib: Production table to use as the grammar prior
                (typically the library state after observing prior melodies).

        Returns:
            A ``(prog_trajs, recon, rate, distortion)`` tuple (same layout
            as :meth:`GreedyDP_PCFGCompressor.run_compose_human`).
        """
        self.lib.production = global_lib

        _, _, prog_trajs = self.run_inner_dp_per_task(task, None)

        # Update the results
        recon = []
        for prog_ind in range(len(prog_trajs)):
            recon += prog_trajs["recon"][prog_ind].tolist()

        return (
            prog_trajs,
            recon,
            -prog_trajs["log_prob"].sum(),
            prog_trajs["distortion"].sum(),
        )

    def run(self, tasks: List[np.array], save_path: str) -> None:
        """
        Main function for sampling programs for all tasks.

        Args:
            tasks (List[np.array]): A list of DataFrames, each containing task information.
            save_path (str): Path to save the sampled programs.
        """
        # Initialize deques to store task-related data
        max_len = self.melody_backtrack_budget + 1
        task_rates = self._construct_deque(max_len, [0])
        task_distortions = self._construct_deque(max_len, [0])
        task_progs = self._construct_deque(max_len, [pd.DataFrame()])
        global_lib_list = self._construct_deque(max_len, [self.init_pm])

        # # TODO: only for debugging
        # tasks = [task[:20] for task in tasks]

        # Process each task
        for n in range(self.task_start_ind, self.task_start_ind + self.task_num):
            print(f"Processing Task {n}/{len(tasks)}", flush=True)

            if n == 0:
                # Handle the first task differently
                rate, distortion, prog_trajs = self.run_inner_dp_per_task(tasks[n])

                # Update the rates, distortions
                task_rates.append([0] * self.melody_backtrack_budget + [rate])
                task_distortions.append(
                    [0] * self.melody_backtrack_budget + [distortion]
                )
                task_progs.append(
                    [pd.DataFrame()] * self.melody_backtrack_budget + [prog_trajs]
                )

                # update library and save to the global library list
                self.lib.update_post_lib(prog_trajs)
                global_lib_list.append(
                    [self.init_pm] * self.melody_backtrack_budget
                    + [self.lib.production]
                )

            else:
                # Process subsequent tasks
                rate, distortion, prog_trajs, global_lib = self.run_per_task(
                    tasks,
                    n,
                    [task_rates, task_distortions, task_progs, global_lib_list],
                )

                # Update task-related deques
                task_rates.append(rate)
                task_distortions.append(distortion)
                task_progs.append(prog_trajs)
                global_lib_list.append(global_lib)

            # ----- Temporary save
            if save_path:
                save_dict = {
                    "task_rates": task_rates[-1],
                    "task_distortions": task_distortions[-1],
                    "task_progs": task_progs[-1],
                    "global_lib_list": global_lib_list[-1],
                }
                self.save_result_per_task(n, save_dict, save_path)

        return task_progs[-1][-1]


class GreedyDP_HAGCompressor(GreedyDP_AGCompressor):
    """Greedy DP compressor with a Hierarchical Adaptor Grammar (global + local).

    Extends :class:`GreedyDP_AGCompressor` with a second, per-task library.
    Within each melody, the local library is updated sub-program by sub-program
    via ``lib.update_local_lib``.  Across melodies, the global library is
    updated via ``lib.update_post_lib``.

    The two-level update schedule means programs that appear repeatedly within
    a single melody receive a stronger prior boost (local update) while programs
    shared across melodies benefit from the global library.
    """

    def __init__(self, program_lib: object, args: object) -> None:
        super().__init__(program_lib, args)

        self.melody_backtrack_budget = args.melody_backtrack_budget
        # For HAG, the canonical "initial" state is the global production table
        self.init_pm = self.lib.global_production
        self.init_table = dict(self.lib.history_table)

    def run_inner_dp_per_task(
        self,
        task: np.ndarray,
        rates: Optional[list] = None,
        distortions: Optional[list] = None,
        prog_list: Optional[list] = None,
        local_lib: Optional[list] = None,
    ) -> Tuple[float, float, pd.DataFrame, pd.DataFrame]:
        """Run the inner greedy DP for one melody with local library updates.

        Performs the same 1-D greedy DP as :class:`GreedyDP_PCFGCompressor`
        but additionally updates the *local* (per-task) library after each
        committed sub-program via ``lib.update_local_lib``.  The local library
        state after the full melody is returned alongside the standard
        rate / distortion / program outputs.

        The inner DP loop:

        1. For each DP position ``i``, sample a program for ``task[i:]``.
        2. Update the local library with the chosen program.
        3. Look back up to ``submelody_backtrack_budget`` notes; if re-segmenting
           from an earlier start yields lower cost, adopt that segmentation
           and update the local library accordingly.
        4. Advance ``i`` by ``recon_len`` of the committed program.

        Args:
            task: Note sequence to compress.
            rates: Unused (signature compatibility with parent); ignored.
            distortions: Unused; ignored.
            prog_list: Unused; ignored.
            local_lib: Unused; ignored.

        Returns:
            A ``(rate, distortion, prog_trajs, local_lib)`` tuple where
            ``local_lib`` is the local production table after processing
            the full melody.
        """
        # Task information
        task_len = len(task)

        # Initialize the rates, distortions, and other structures
        note_window = self.submelody_backtrack_budget
        rates = self._construct_deque(note_window + 1, 0)
        distortions = self._construct_deque(note_window + 1, 0)
        prog_list = self._construct_deque(note_window + 1, pd.DataFrame())
        # local_lib_list = self._construct_deque(note_window + 1, self.init_pm)
        local_lib_list = self._construct_deque(note_window + 1, self.lib.init_pm)
        start_inds = self._construct_deque(note_window + 1, float(-np.inf))

        # Iterate from the start of the sequence to the end
        i = 0
        while i < len(task):
            start_inds.append(i)

            # Update local library with current production state
            self.lib.production = local_lib_list[-1]

            # Perform a single forward simulation step
            frame, first_filled = self._run_forward_simulation_one_step(task[i:])
            self.lib.update_local_lib(first_filled)
            best_local_lib = self.lib.production

            # Concatenate new program with the previous one to get new rate and distortion
            min_prog = pd.concat([prog_list[-1], first_filled], ignore_index=True)
            cur_recon_len = min(first_filled["recon_len"].values[0], task_len - i)
            i += cur_recon_len

            # Calculate initial cost
            min_cost = self._calculate_cost(min_prog)

            for l in range(1, note_window + 1):
                start_ind = start_inds[-l]

                if start_ind < 0:
                    break

                # Calculate reward for compressing the submelody sequence[i-l:i]
                subtask = task[start_ind:i]

                # Intialize the local library
                self.lib.production = local_lib_list[-l]

                # A function that computes reward
                new_prog = self._run_forward_simulation_multi_step(subtask)
                new_combined_prog = pd.concat(
                    [prog_list[-l], new_prog], ignore_index=True
                )
                new_cost = self._calculate_cost(new_combined_prog)

                # Update minimum cost and progression if applicable
                if new_cost < min_cost:
                    min_cost = new_cost
                    min_prog = new_combined_prog

                    # Update the local library and the list
                    for prog_ind in range(len(new_prog)):
                        self.lib.update_local_lib(new_prog.iloc[[prog_ind]])
                    best_local_lib = self.lib.production

            # Update dynamic programming structures
            rates.append(-min_prog["log_prob"].sum())
            distortions.append(min_prog["distortion"].sum())
            prog_list.append(min_prog)
            local_lib_list.append(best_local_lib)

        return rates[-1], distortions[-1], prog_list[-1], local_lib_list[-1]

    def run_per_task(
        self,
        tasks: List[np.ndarray],
        cur_task_ind: int,
        cache: list,
    ) -> tuple:
        """Outer greedy DP for task ``cur_task_ind`` with two-level library updates.

        Extends :meth:`GreedyDP_AGCompressor.run_per_task` by tracking both
        local (per-melody) and global library states in the cache.  For each
        lookback window ``l``, the outer DP:

        1. Restores the global library and history table from the cache at
           position ``cur_task_ind - l``.
        2. Re-runs the inner DP (with local updates) for melodies
           ``cur_task_ind - l`` through ``cur_task_ind``.
        3. Updates both local and global libraries and stores the new states.
        4. Computes the cumulative RD cost and keeps the window with minimum cost.

        Args:
            tasks: Full list of melody arrays.
            cur_task_ind: Index of the melody currently being processed.
            cache: A 6-element list of rolling deques:
                ``[rates, distortions, progs, global_lib, global_table, local_lib]``.

        Returns:
            A tuple ``(rates, distortions, progs, global_libs, global_tables,
            local_libs)`` — each a list sliced to the last ``melody_backtrack_budget``
            entries for the winning window.
        """
        melody_backtrack_budget = self.melody_backtrack_budget
        (
            cache_rates,
            cache_distortions,
            cache_progs,
            cache_global_lib,
            cache_global_table,
            cache_local_lib,
        ) = cache

        min_cost = float("inf")
        # Try all possible submelody lengths that can end at index i
        for l in range(0, min(melody_backtrack_budget, cur_task_ind) + 1):
            # Extract relevant information from the cache for comparison RD cost
            temp_ind = -(l + 1)
            temp_progs = list(cache_progs[temp_ind])
            temp_rates = list(cache_rates[temp_ind])
            temp_distortions = list(cache_distortions[temp_ind])

            temp_local_libs = list(cache_local_lib[temp_ind])
            temp_global_libs = list(cache_global_lib[temp_ind])
            temp_global_tables = list(cache_global_table[temp_ind])

            # Update the global library
            self.lib.global_production = pd.DataFrame(temp_global_libs[-1])
            self.lib.history_table = dict(temp_global_tables[-1])

            for n in range(cur_task_ind - l, cur_task_ind + 1):
                task = tasks[n]

                # A function that computes reward
                (
                    cur_rate,
                    cur_dist,
                    cur_prog,
                    cur_local_lib,
                ) = self.run_inner_dp_per_task(task)

                temp_progs.append(cur_prog)
                temp_rates.append(cur_rate)
                temp_distortions.append(cur_dist)
                temp_local_libs.append(cur_local_lib)

                # Update the library
                self.lib.update_post_lib(cur_prog, cur_local_lib)
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

        return (
            min_rate[-melody_backtrack_budget - 1 :],
            min_distortion[-melody_backtrack_budget - 1 :],
            min_prog_traj[-melody_backtrack_budget - 1 :],
            min_global_lib[-melody_backtrack_budget - 1 :],
            min_global_table[-melody_backtrack_budget - 1 :],
            min_local_lib[-melody_backtrack_budget - 1 :],
        )

    def run_submelody_human(
        self,
        tasks: List[pd.DataFrame],
        part_id: int,
        save_path: str,
        mask_data=None,
    ) -> None:
        """
        This is the main function for sampling programs for all tasks
        Args:
            tasks: a list of dataframes, each of which contains the task information
            save: whether to save the sampled programs
        """
        progs = []
        global_libs = []
        local_libs = []
        results = {"recon": [], "rate": [], "distortion": []}

        self.lib.production = self.init_pm
        self.lib.global_production = self.init_pm
        self.lib.history_table = {}
        for type_string in self.lib.type_strings:
            self.lib.history_table[type_string] = 0

        for i in range(NUM_MELODY):
            print(f"Task {i}/{NUM_MELODY}", flush=True)
            self.lib.production = self.init_pm

            (
                progs_per_task,
                recon_per_task,
                rate_per_task,
                distortion_per_task,
                loc_libs_per_task,
            ) = ([], [], [], [], [])

            for j in range(NUM_SUBMELODY - 1):
                cur_task = tasks[i][j]
                _, _, prog_trajs = GreedyDP_AGCompressor.run_inner_dp_per_task(
                    self, cur_task, mask_data[i][j] if mask_data is not None else None
                )
                # Update the library using the last submelody
                for prog_ind in range(len(prog_trajs)):
                    one_prog = prog_trajs.iloc[[prog_ind]]
                    self.lib.update_local_lib(one_prog)

                progs_per_task.append(prog_trajs)
                loc_libs_per_task.append(self.lib.production)

                # Update the results
                recon = []
                for prog_ind in range(len(prog_trajs)):
                    recon += prog_trajs["recon"][prog_ind].tolist()
                recon_per_task.append(recon)
                rate_per_task.append(-prog_trajs["log_prob"].sum())
                distortion_per_task.append(prog_trajs["distortion"].sum())

            # Not update the global libaray using the last submelody
            for j in range(NUM_SUBMELODY - 1):
                cur_prog_list = progs_per_task[j]
                for prog_ind in range(len(cur_prog_list)):
                    one_prog = cur_prog_list.iloc[[prog_ind]]
                    self.lib.update_post_lib(one_prog, self.lib.production)

            local_libs.append(loc_libs_per_task)
            global_libs.append(self.lib.global_production)
            progs.append(progs_per_task)
            results["recon"].append(recon_per_task)
            results["rate"].append(rate_per_task)
            results["distortion"].append(distortion_per_task)

        if save_path != None:
            self.save_result_per_task(f"p{part_id}", {"prog_trajs": progs}, save_path)
            self.save_result_per_task(f"p{part_id}", {"results": results}, save_path)
            self.save_result_per_task(
                f"p{part_id}", {"global_libs": global_libs}, save_path
            )
            self.save_result_per_task(
                f"p{part_id}", {"local_libs": local_libs}, save_path
            )

        return progs, results

    def run_compose_human(
        self,
        task: np.ndarray,
        local_lib: pd.DataFrame,
        global_lib: pd.DataFrame,
        global_table: dict,
    ) -> Tuple[pd.DataFrame, list, float, float]:
        """Run the HAG compressor on a single melody given fixed library states.

        Loads the provided local, global, and history-table states into the
        grammar before running the inner DP, conditioning predictions on a
        specific library snapshot (e.g., after observing prior melodies for
        a participant).

        Args:
            task: Note sequence to compress.
            local_lib: Per-task production table to use as the local prior.
            global_lib: Global production table to use as the global prior.
            global_table: History table for the global Pitman-Yor process.

        Returns:
            A ``(prog_trajs, recon, rate, distortion)`` tuple.
        """
        results = {"recon": [], "rate": [], "distortion": []}

        self.lib.production = local_lib
        self.lib.global_production = global_lib
        self.lib.history_table = global_table

        _, _, prog_trajs = GreedyDP_AGCompressor.run_inner_dp_per_task(self, task, None)

        # Update the results
        recon = []
        for prog_ind in range(len(prog_trajs)):
            recon += prog_trajs["recon"][prog_ind].tolist()

        return (
            prog_trajs,
            recon,
            -prog_trajs["log_prob"].sum(),
            prog_trajs["distortion"].sum(),
        )

    def run(self, tasks: List[np.array], save_path: str) -> None:
        """
        Main function for sampling programs for all tasks.

        Args:
            tasks (List[pd.DataFrame]): A list of DataFrames, each containing task information.
            save_path (str): Path to save the results of the sampled programs.

        Notes:
            - This method iterates through tasks, applying inner and outer loops to generate
            and refine programs for each task based on provided parameters.
            - The process involves maintaining and updating local/global libraries and
            associated metadata for each task.
        """
        # Maximum allowed length for deque-based storage
        max_len = self.melody_backtrack_budget + 1

        # Initialize deques for various metrics and libraries
        task_rates = self._construct_deque(max_len, [0])
        task_distortions = self._construct_deque(max_len, [0])
        task_progs = self._construct_deque(max_len, [pd.DataFrame()])
        local_lib_list = self._construct_deque(max_len, [self.init_pm])

        global_lib_list = self._construct_deque(max_len, [self.init_pm])
        global_table_list = self._construct_deque(max_len, [dict(self.init_table)])

        # # TODO: only for debugging
        # tasks = [task[:15] for task in tasks]

        # ----- Outer loop: iteration over all tasks
        for n in range(self.task_start_ind, self.task_start_ind + self.task_num):
            # Task information
            print(f"Task {n}/{len(tasks)}", flush=True)

            if n == 0:
                # Initialize cache for the first task
                rate, distortion, prog_trajs, loc_lib = self.run_inner_dp_per_task(
                    tasks[n]
                )

                # Update the rates, distortions
                task_rates.append([0] * self.melody_backtrack_budget + [rate])
                task_distortions.append(
                    [0] * self.melody_backtrack_budget + [distortion]
                )
                task_progs.append(
                    [pd.DataFrame()] * self.melody_backtrack_budget + [prog_trajs]
                )
                local_lib_list.append(
                    [self.init_pm] * self.melody_backtrack_budget + [loc_lib]
                )

                # update library and save to the global library list
                self.lib.update_post_lib(prog_trajs, loc_lib)
                global_lib_list.append(
                    [self.init_pm] * self.melody_backtrack_budget
                    + [self.lib.global_production]
                )
                global_table_list.append(
                    [dict(self.init_table)] * self.melody_backtrack_budget
                    + [dict(self.lib.history_table)]
                )

            else:
                # Find programs that are consistent with data, and compute the likelihood of each program
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

            # ----- Temporary save
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

        return (
            task_progs[-1][-1],
            local_lib_list[-1],
            global_lib_list[-1],
            global_table_list[-1],
        )
