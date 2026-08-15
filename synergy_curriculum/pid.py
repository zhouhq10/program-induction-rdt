"""
Partial information decomposition for program libraries.

Provenance
----------
The synergy estimator here is the one recovered from commit `adddda5`
("Finish the synergy computation.", 2024-01-03), originally
`compression/program/pid_calculator.py` and vendored into this file as
`PIDCalculator`. That class only ever implemented
`synergy()`; `redundancy()`, `unique()`, `mutual()` and `decomposition()` were
called by the old notebooks but never written (which is why they are commented
out in `exp/archive/4_pid.ipynb`). They are completed here following
Williams & Beer (2010), reusing the `_Imin` / `_Imax` helpers that
`pid_helpers.py` already provides.

Measures, for sources X_1..X_n and target y:

    mutual      I(X; y)                                  -- total information
    redundancy  Imin over the singleton sources          -- shared information
    unique_i    I(X_i; y) - redundancy                   -- carried by X_i alone
    synergy     I(X; y) - Imax over the (n-1)-subsets    -- only in combination

Synergy is the quantity the curriculum search maximises: a high-synergy library
is one whose programs predict task outcome *in combination* but not
individually.
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.append("..")

from synergy_curriculum.pid_helpers import (
    _compute_mutual_info,
    _compute_specific_info,
    _conditional_probability_from_joint,
    _Imax,
    _Imin,
    _isbinary,
    _joint_probability,
    _joint_sub,
    lazy_property,
)

# `_map_binary_par` packs each row of a binary X into one integer, and the
# result is cast to int64. Beyond 62 sources that silently overflows.
MAX_SOURCES = 62


class PIDCalculator:
    """
    https://github.com/pietromarchesi/pidpy

    Calculator class for the partial information decomposition of mutual
    information for discrete variables.

    Recovered verbatim from commit `adddda5`; only `synergy()` was ever
    implemented. `PID` below subclasses it and supplies the rest. Kept as its
    own class so the recovered estimator stays untouched.

    Parameters
    ----------
    X : 2D ndarray, shape (n_samples, n_features)
        Data from which to obtain the information decomposition.

    y : 1D ndarray, shape (n_samples, )
        Array containing the labels of the dependent variable.

    **kwargs
        safe_labels : bool, optional (default = False)
            If `True`, it is assumed that the `n` labels the compose `y`
            are the integers in `range(n)`. If `False`, the above is checked
            and if it is found to be false, `y` is mapped so that the label
            values are `range(n)`. This is necessary because the label values
            are used for indexing in the construction of probability tables.
            `safe_labels` should be kept to `False`, setting to `True` is only
            done internally to speed up the initialization of the PID calculators
            used to generate surrogate data.

        binary : bool, optional
            Directly specifies whether `X` is a binary array. If not provided,
            the PID calculator runs a check on the whole array `X` to verify
            if it is binary. This parameter is used internally and passed
            to the surrogates to avoid multiple checks being run on the same
            array.
            If the data is binary and has a relatively low number of variables,
            the probability tables can be generated with a faster routine.

        labels : list, optional
            Allows to directly specify the labels in `y`. If not provided,
            labels are extracted from the `y` array using `set`. This parameter
            is used internally to speed up the instantiation of surrogate
            calculators.

        n_jobs : int, optional
            The generation of surrogate data sets and the computation of
            their information values can be executed in parallel, with n_jobs
            specifying the number of parallel jobs to be launched through
            the Joblib library. There is a known issue with large arrays,
            for which parallelization is not possible. In that case a
            execution is continued with 1 sequential job.

        alpha: float, optional
            If the `decomposition` method is asked to return the sum of the
            unique information while `test_significance` is set to true,
            the sum is of the unique information of individual sources is
            computed only for sources which are significant with significance
            level `alpha`. Default level is `0.01`.


    Notes
    -----
    Partial information decomposition of binary data (`X` contains only
    `0` and `1`, no restrictions on `y`) is supported for an arbitrary
    number of variables (although ensure that the number of data points is
    sufficient to accurately build the probability tables).
    Decomposition of integer non-binary data is only supported for up to
    three variables.
    """

    def __init__(self, X, y, **kwargs):
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "The number of samples in the feature and labels" "arrays should match."
            )

        if not issubclass(X.dtype.type, np.integer):
            X = X.astype("int")

        if not "binary" in kwargs:
            self.binary = _isbinary(X)
        else:
            self.binary = kwargs["binary"]

        if not "n_jobs" in kwargs:
            self.n_jobs = -1
        else:
            self.n_jobs = kwargs["n_jobs"]

        if not "alpha" in kwargs:
            self.alpha = 0.05
        else:
            self.alpha = kwargs["alpha"]

        if not self.binary and X.shape[1] > 3:
            raise NotImplementedError(
                "Decomposition of non-binary data with more"
                "than 3 variables is not supported yet."
            )

        # labels are passed by the main calculator as kwargs to the calculators
        # used for debiasing, to avoid recomputing the set of labels
        if not "labels" in kwargs:
            original_labels = list(set(y))
        else:
            original_labels = kwargs["labels"]

        if not ("safe_labels" in kwargs and kwargs["safe_labels"]):
            if not original_labels == range(len(original_labels)):
                y = np.array([original_labels.index(lab) for lab in y])

        self.labels = range(len(original_labels))
        self.original_labels = original_labels

        self.Nlabels = len(self.labels)
        self.verbosity = 0
        self.X = X
        self.y = y
        self.Nsamp = y.shape[0]
        self.Nneurons = X.shape[1]
        self.surrogate_pool = []

    @lazy_property
    def y_mar_(self):
        """
        Marginal probability of `y`.
        """
        # TODO build this without referring to the full joint
        y_mar_ = self.joint_full_.sum(axis=0)
        return y_mar_

    @lazy_property
    def X_mar_(self):
        """
        Marginal probability of `X`.
        """
        X_mar_ = self.joint_full_.sum(axis=1)
        return X_mar_

    @lazy_property
    def joint_full_(self):
        """
        Full joint probability of `X` and `y`.
        """
        joint_full_ = _joint_probability(self.X, self.y, binary=self.binary)
        return joint_full_

    @lazy_property
    def mi_full_(self):
        """
        Mutual of information between `X` and `y`.
        """
        mi_full_ = _compute_mutual_info(self.X_mar_, self.y_mar_, self.joint_full_)
        return mi_full_

    @lazy_property
    def joint_sub_(self):
        """
        Joint probability tables of all groups of `n-1` variables of `X`.
        """
        joint_sub_ = _joint_sub(self.X, self.y, binary=self.binary)
        return joint_sub_

    @lazy_property
    def spec_info_sub_(self):
        """
        Specific information for every label for all groups of `n-1` variables
        of `X`.
        """
        spec_info_sub_ = self._spec_info(self.labels, self.joint_sub_)
        return spec_info_sub_

    def _spec_info(self, labels, joints):
        spec_info_full_ = []
        for lab in labels:
            spec_info_lab = []
            for joint in joints:
                cond_Xy, cond_yX = _conditional_probability_from_joint(joint)
                info = _compute_specific_info(lab, self.y_mar_, cond_Xy, cond_yX, joint)
                spec_info_lab.append(info)
            spec_info_full_.append(spec_info_lab)
        return spec_info_full_

    def synergy(self, debiased=False, n=50):
        """
        Compute the pure synergy between the variables of the data array X.

        Parameters
        ----------
        debiased: bool, optional
            If True, synergy is debiased with shuffled surrogates. Default is
            False.

        n : int, optional
            Number of surrogate data sets to be used for debiasing. Defaults
            to 50.

        Returns
        -------
        synergy : float
            Pure synergy of the variables in `X`.
        standard_deviation : float
            Standard deviation of the synergy of the surrogate sets, only
            returned if `debiased = True`.
        """

        if debiased:
            # `_debiased` is supplied by `PID`; the recovered class never had it.
            self.syn = self._debiased("synergy", n)
        else:
            self.syn = self.mi_full_ - _Imax(self.y_mar_, self.spec_info_sub_)
        return self.syn


@dataclass
class Decomposition:
    """Result of a full partial information decomposition."""

    mutual: float
    redundancy: float
    synergy: float
    unique: np.ndarray
    names: Optional[List[str]] = None
    surrogate_mean: Dict[str, float] = field(default_factory=dict)
    surrogate_std: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, float]:
        d = {
            "mutual": self.mutual,
            "redundancy": self.redundancy,
            "synergy": self.synergy,
            "unique_sum": float(np.sum(self.unique)),
        }
        for measure, mean in self.surrogate_mean.items():
            d[f"{measure}_surrogate_mean"] = mean
            d[f"{measure}_surrogate_std"] = self.surrogate_std.get(measure, np.nan)
        return d

    def __str__(self) -> str:
        lines = [
            f"  mutual      I(X;y)  = {self.mutual: .4f} bits",
            f"  redundancy          = {self.redundancy: .4f} bits",
            f"  synergy             = {self.synergy: .4f} bits",
            f"  unique (sum)        = {np.sum(self.unique): .4f} bits",
        ]
        for measure in ("mutual", "redundancy", "synergy"):
            if measure in self.surrogate_mean:
                observed = getattr(self, measure)
                mean = self.surrogate_mean[measure]
                std = self.surrogate_std.get(measure, float("nan"))
                z = (observed - mean) / std if std else float("nan")
                lines.append(
                    f"  {measure:<10} debiased = {observed - mean: .4f} bits "
                    f"(shuffled {mean: .4f} +/- {std:.4f}, z = {z:.2f})"
                )
        if self.names is not None:
            order = np.argsort(self.unique)[::-1][:5]
            lines.append("  top unique sources:")
            lines += [f"    {self.unique[i]: .4f}  {self.names[i]}" for i in order]
        return "\n".join(lines)


class PID(PIDCalculator):
    """
    Partial information decomposition of a binary occurrence matrix.

    Parameters
    ----------
    X : (n_samples, n_sources) int array
        Binary occurrence matrix -- X[i, j] = 1 if program j occurs in the
        compression of melody i.
    y : (n_samples,) int array
        Discrete outcome label per melody (see `occurrence.discretize`).
    binary : bool
        Whether X is binary. Keep True; the non-binary path in the underlying
        helpers supports at most 3 sources and needs `pymorton`.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, binary: bool = True, **kwargs):
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_samples, n_sources), got {X.shape}")
        if X.shape[1] < 2:
            raise ValueError(
                "PID needs at least 2 sources; synergy is undefined for a "
                "single variable."
            )
        if X.shape[1] > MAX_SOURCES:
            raise ValueError(
                f"{X.shape[1]} sources exceeds the {MAX_SOURCES}-source limit of "
                "the binary row-packing in pid_helpers._map_binary_par."
            )
        super().__init__(X, y, binary=binary, **kwargs)

    # ----- Probability tables for the singleton sources -----
    # The recovered class builds `joint_sub_` (all (n-1)-subsets, used by
    # synergy). Redundancy and unique information need the singletons instead.
    @lazy_property
    def joint_indiv_(self):
        return [
            _joint_probability(self.X[:, [i]], self.y, binary=self.binary)
            for i in range(self.Nneurons)
        ]

    @lazy_property
    def spec_info_indiv_(self):
        return self._spec_info(self.labels, self.joint_indiv_)

    @lazy_property
    def mi_indiv_(self):
        return np.array(
            [
                _compute_mutual_info(joint.sum(axis=1), self.y_mar_, joint)
                for joint in self.joint_indiv_
            ]
        )

    # ----- Measures -----
    def mutual(self, debiased: bool = False, n: int = 50) -> float:
        """Total mutual information I(X; y) carried by all sources jointly."""
        if debiased:
            return self._debiased("mutual", n)
        return float(self.mi_full_)

    def redundancy(self, debiased: bool = False, n: int = 50) -> float:
        """Information available redundantly from any single source (Imin)."""
        if debiased:
            return self._debiased("redundancy", n)
        return float(_Imin(self.y_mar_, self.spec_info_indiv_))

    def unique(self, debiased: bool = False, n: int = 50) -> np.ndarray:
        """Per-source unique information, I(X_i; y) - redundancy."""
        if debiased:
            return self._debiased("unique", n)
        return np.asarray(self.mi_indiv_ - self.redundancy(), dtype=float)

    # `synergy()` is inherited verbatim from the recovered PIDCalculator:
    #     synergy = mi_full_ - Imax(y_mar_, spec_info_sub_)

    def decomposition(
        self,
        debiased: bool = False,
        n: int = 50,
        names: Optional[Sequence[str]] = None,
    ) -> Decomposition:
        """Compute every measure at once."""
        dec = Decomposition(
            mutual=self.mutual(),
            redundancy=self.redundancy(),
            synergy=float(self.synergy()),
            unique=self.unique(),
            names=list(names) if names is not None else None,
        )
        if debiased:
            for measure in ("mutual", "redundancy", "synergy"):
                mean, std = self.surrogate_stats(measure, n)
                dec.surrogate_mean[measure] = mean
                dec.surrogate_std[measure] = std
        return dec

    # ----- Debiasing against shuffled surrogates -----
    def surrogate_stats(self, measure: str, n: int = 50) -> tuple:
        """
        Mean and sd of `measure` over `n` surrogates with y shuffled.

        Shuffling y destroys any X-y relationship, so the surrogate mean is the
        value the estimator returns from finite-sample bias alone.
        """
        rng = np.random.default_rng(getattr(self, "_surrogate_seed", 0))
        values = []
        for _ in range(n):
            y_shuffled = rng.permutation(self.y)
            surrogate = PID(self.X, y_shuffled, binary=self.binary)
            value = getattr(surrogate, measure)()
            values.append(np.sum(value) if np.ndim(value) else value)
        values = np.asarray(values, dtype=float)
        return float(values.mean()), float(values.std())

    def _debiased(self, measure: str, n: int = 50):
        """Observed value minus the shuffled-surrogate mean."""
        observed = getattr(self, measure)()
        mean, std = self.surrogate_stats(measure, n)
        self.surrogate_std_ = std
        return observed - mean


# ----- Convenience wrappers -----
def synergy(X: np.ndarray, y: np.ndarray, debiased: bool = False, n: int = 50) -> float:
    """Synergy of a binary occurrence matrix. Returns -inf on a degenerate X."""
    X = np.asarray(X)
    # A source that never varies contributes nothing and makes the (n-1)-subset
    # tables singular; a constant y has no information to decompose.
    if X.shape[1] < 2 or len(set(np.asarray(y).tolist())) < 2:
        return float("-inf")
    try:
        with np.errstate(invalid="ignore", divide="ignore"):
            # Unobserved occurrence patterns give 0/0 in
            # `_conditional_probability_from_joint`. The resulting NaNs are
            # skipped downstream by the `c > 1e-8` guard in
            # `_compute_specific_info`, so the warning is noise.
            pid = PID(X, y)
            return float(
                pid.synergy(debiased=debiased, n=n) if debiased else pid.synergy()
            )
    except (ValueError, IndexError, ZeroDivisionError):
        return float("-inf")


def check_sample_adequacy(X: np.ndarray, y: np.ndarray, verbose: bool = True) -> dict:
    """
    Flag the undersampling that makes a synergy estimate meaningless.

    With n binary sources there are 2^n possible occurrence patterns. When
    almost every melody has its own distinct pattern, I(X; y) saturates at the
    entropy of y and the decomposition is measuring sample size, not structure.
    This was a live problem in the original notebooks (30 sources, ~100 rows).
    """
    X = np.asarray(X)
    n_samples, n_sources = X.shape
    n_patterns = len({tuple(row) for row in X})
    saturation = n_patterns / max(n_samples, 1)
    n_labels = len(set(np.asarray(y).tolist()))

    report = {
        "n_samples": n_samples,
        "n_sources": n_sources,
        "n_distinct_patterns": n_patterns,
        "saturation": saturation,
        "n_labels": n_labels,
        "samples_per_source": n_samples / max(n_sources, 1),
    }

    if verbose:
        if saturation > 0.9:
            warnings.warn(
                f"{n_patterns}/{n_samples} occurrence patterns are distinct "
                f"(saturation {saturation:.2f}). I(X;y) is nearly saturated, so "
                "synergy is dominated by finite-sample bias. Use fewer sources, "
                "more melodies, or compare against debiased=True.",
                stacklevel=2,
            )
        if n_samples < 10 * n_sources:
            warnings.warn(
                f"{n_samples} melodies for {n_sources} sources "
                f"({report['samples_per_source']:.1f} per source). Estimates "
                "will be noisy; treat the ranking as heuristic.",
                stacklevel=2,
            )
    return report


def synergy_dit(X: np.ndarray, y: np.ndarray, measure: str = "PID_WB"):
    """
    Optional cross-check against the `dit` package (`pip install dit`).

    UNTESTED: `dit` is not installed in this environment, so this path has
    never been run. It is here as a starting point for validating the in-repo
    estimator on a small number of sources, not as a drop-in replacement --
    `dit`'s exact decompositions are exponential in the number of sources and
    are only practical for roughly <= 5.
    """
    try:
        import dit
        from dit import pid as dit_pid
    except ImportError as exc:  # pragma: no cover
        raise ImportError("synergy_dit requires `pip install dit`") from exc

    X = np.asarray(X)
    outcomes = {}
    for row, label in zip(X, np.asarray(y)):
        key = tuple(int(v) for v in row) + (int(label),)
        outcomes[key] = outcomes.get(key, 0) + 1
    total = sum(outcomes.values())
    d = dit.Distribution(
        [list(map(str, k)) for k in outcomes],
        [c / total for c in outcomes.values()],
    )
    decomposition = getattr(dit_pid, measure)(d)
    return decomposition
