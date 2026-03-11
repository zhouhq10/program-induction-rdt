import os
import random
import pickle
import argparse
import numpy as np
from scipy.stats import wilcoxon, rankdata


def extract_obj(file_path: str) -> object:
    """Load a pickled object from disk.

    Args:
        file_path: Path to the ``.obj`` pickle file.

    Returns:
        The deserialised Python object.
    """
    with open(file_path, "rb") as file:
        obj_file = pickle.load(file)
    return obj_file


def save_obj(obj, file_path):
    with open(file_path, "wb") as file:
        pickle.dump(obj, file)


def create_save_path(args: argparse.Namespace) -> str:
    """Build and create the output directory for this experimental run.

    The directory path is constructed from a user-specified list of argument
    names (``--folder_name``), so that different hyper-parameter settings are
    automatically stored in separate sub-folders.

    Args:
        args: Parsed command-line arguments.  Must contain ``save_path``,
            ``experiname``, and ``folder_name`` (a Python-literal list string
            of argument names, e.g. ``"['beta', 'random_seed']"``).

    Returns:
        Absolute path of the created output directory.
    """
    folder_name_list = eval(args.folder_name)
    folder_name = "_".join(
        [f"{name}_{getattr(args, name)}" for name in folder_name_list]
    )
    full_folder_path = os.path.join(args.save_path, args.experiname, folder_name)
    os.makedirs(full_folder_path, exist_ok=True)
    return full_folder_path


def prepare_task_data(
    task_path: str,
    random_seed: int,
    random_seq: int,
    num: int = 50,
) -> list:
    """Load melody task data and prepare a reproducible shuffled subset.

    Two random seeds are used:
      - ``random_seed`` controls which ``num`` tasks are sampled from the full
        dataset (reproducible sub-sampling).
      - ``random_seq`` controls the presentation order of those tasks
        (reproducible shuffling).

    Args:
        task_path: Path to a pickle file containing a list of melody sequences
            (each sequence is a 1-D array of note integers in {1, …, 6}).
        random_seed: Seed for sub-sampling tasks from the full dataset.
        random_seq: Seed for shuffling the sampled tasks.
        num: Number of tasks to use.

    Returns:
        List of ``num`` numpy arrays, each representing one melody task.
    """
    task_data = extract_obj(task_path)
    task_data = [np.array(x) for x in task_data]

    rng = random.Random(random_seed)
    task_data = rng.sample(task_data, num)

    rng = random.Random(random_seq)
    rng.shuffle(task_data)

    return task_data


def flatten(xss):
    return [x for xs in xss for x in xs]


def wilcoxon_effect_ci(x, y, n_boot=5000, alternative="two-sided", random_state=None):
    """
    Compute Wilcoxon test, effect size (rank-biserial correlation),
    and bootstrap confidence intervals.

    Parameters:
        x, y : array-like
            Paired data.
        n_boot : int
            Number of bootstrap samples.
        alternative : "two-sided", "less", or "greater"
        random_state : int or None

    Returns:
        dict with:
        - stat
        - pvalue
        - effect_size
        - effect_ci (tuple)
        - median_diff
        - median_ci (tuple)
    """

    x = np.asarray(x)
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)

    # ----- Wilcoxon test -----
    stat, p = wilcoxon(x, y, alternative=alternative)

    # ----- Effect size: rank-biserial correlation -----
    def rbc(a, b):
        diff = a - b
        diff = diff[diff != 0]  # ignore zero differences
        if len(diff) == 0:
            return 0.0
        ranks = rankdata(abs(diff))
        W_pos = ranks[diff > 0].sum()
        W_neg = ranks[diff < 0].sum()
        return (W_pos - W_neg) / (W_pos + W_neg)

    effect = rbc(x, y)

    # ----- Bootstrap CIs -----
    boot_effect = []
    boot_median = []

    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        xb, yb = x[idx], y[idx]
        boot_effect.append(rbc(xb, yb))
        boot_median.append(np.median(xb - yb))

    # Confidence intervals
    eff_ci = (np.percentile(boot_effect, 2.5), np.percentile(boot_effect, 97.5))

    diff = x - y
    med_ci = (np.percentile(boot_median, 2.5), np.percentile(boot_median, 97.5))

    return {
        "stat": stat,
        "pvalue": p,
        "effect_size": effect,
        "effect_ci": eff_ci,
        "median_diff": np.median(diff),
        "median_ci": med_ci,
    }
