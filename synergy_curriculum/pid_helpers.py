import numpy as np
import math
import warnings

from math import log2, fabs


def _isbinary(X):
    return set(X.flatten()) == {0, 1}


def _Imin(y_mar_, spec_info):
    Im = 0
    for i in range(len(y_mar_)):
        Im += y_mar_[i] * np.min(spec_info[i])
    return Im


def _Imax(y_mar_, spec_info):
    Im = 0
    for i in range(len(y_mar_)):
        Im += y_mar_[i] * np.max(spec_info[i])
    return Im


def lazy_property(fn):
    """
    Decorator used to ensure that relevant probability tables
    and other attributes are not recomputed if they have already
    been calculated.
    """
    attr_name = "_lazy_" + fn.__name__

    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            # if getattr(self, 'verbosity') > 0:
            #    print('Computing %s' %fn.__name__)
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazyprop


def _map_nonbinary_array(X):
    """
    Array of integer non-binary values to vector of
    integers using Morton encoding of every row of the input
    array. Supports only up to three variables.
    """
    # TODO: this is fine but every time the value error is raised pymorton prints
    # stuff, which we don't want.
    # Xmap = np.zeros(X.shape[0], dtype = int)
    # for i in range(X.shape[0]):
    #     if X.shape[1] == 3:
    #         try:
    #             Xmap[i] = pymorton.interleave(X[i,0], X[i,1], X[i,2])
    #         except ValueError:
    #             Xmap[i] = pymorton.interleave(int(X[i,0]), int(X[i,1]), int(X[i,2]))
    #     if X.shape[1] == 2:
    #         try:
    #             Xmap[i] = pymorton.interleave(X[i,0], X[i,1])
    #         except ValueError:
    #             Xmap[i] = pymorton.interleave(int(X[i, 0]), int(X[i, 1]))
    # return Xmap

    try:
        import pymorton
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Non-binary decomposition requires the optional `pymorton` package "
            "(pip install pymorton). Binary occurrence matrices, which is what "
            "the curriculum search uses, do not need it."
        ) from exc

    Xmap = np.zeros(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
        if X.shape[1] == 3:
            Xmap[i] = pymorton.interleave(int(X[i, 0]), int(X[i, 1]), int(X[i, 2]))
        if X.shape[1] == 2:
            Xmap[i] = pymorton.interleave(int(X[i, 0]), int(X[i, 1]))
    return Xmap


def _map_binary_par(x, n):
    """
    Takes a binary array x and converts it into an integer by interpreting the binary digits of the array as the binary representation of an integer. It accumulates the result as it iterates through the binary digits, applying the correct place values for each set bit.
    """
    tot = 0
    p = 1

    for i in range(n):
        if x[i]:
            tot += p
        p = p << 1

    return tot


def _map_binary_array_par_inner(X, Xmap, N, n):
    for i in range(N):
        Xmap[i] = _map_binary_par(X[i, :], n)
    return Xmap


def _map_binary_array_par(X):
    N = X.shape[0]
    n = X.shape[1]
    Xmap = [0] * N

    Xmapout = _map_binary_array_par_inner(X, Xmap, N, n)

    return np.array(Xmapout, dtype="int64")


def _map_array(X, binary=True):
    """
    High-level mapping function used in PIDCalculator.
    """
    if binary:
        Xmap = _map_binary_array_par(X)
    else:
        Xmap = _map_nonbinary_array(X)
    return Xmap


def _compute_joint_probability_nonbin(X, y):
    nsamp = y.shape[0]
    nlabels = len(set(y))
    vals = np.array(sorted(list(set(X))))
    nvals = vals.shape[0]
    joint = np.zeros((nvals, nlabels), dtype=np.int64)

    for i in range(nsamp):
        for j in range(nvals):
            if X[i] == vals[j]:
                ind = j
        joint[ind, y[i]] += 1

    return joint / float(nsamp)


def _compute_joint_probability_bin(X, y, nvals):
    """
    Args:
        nvals: the number of unique values or categories that can exist in the X array.
    """
    nsamp = y.shape[0]
    nlabels = len(set(y))
    joint = np.zeros((nvals, nlabels), dtype=np.int64)

    for i in range(nsamp):
        joint[X[i], y[i]] += 1

    return joint / float(nsamp)


def _joint_probability(X, y, binary=True):
    """
    Computes the joint probability of X and y.
    """
    if X.ndim > 1:
        Xmap = _map_array(X, binary=binary)
        N = X.shape[1]
    else:
        Xmap = X
        N = 1

    if binary and N < 12:
        nvals = 2**N
        joint = _compute_joint_probability_bin(Xmap, y, nvals)
    else:
        joint = _compute_joint_probability_nonbin(Xmap, y)

    return joint


def _compute_mutual_info(X_mar_, y_mar_, joint):
    """
    Computes the mutual information between X and y.
    """
    I = 0
    xlen = X_mar_.shape[0]
    ylen = y_mar_.shape[0]

    for i in range(xlen):
        for j in range(ylen):
            if fabs(joint[i, j]) > 1e-8:
                I += joint[i, j] * log2(joint[i, j] / (X_mar_[i] * y_mar_[j]))

    return I


def _conditional_probability_from_joint(joint):
    X_mar = joint.sum(axis=1)
    y_mar = joint.sum(axis=0)

    cond_Xy = joint.astype(float) / y_mar[np.newaxis, :]
    cond_yX = joint.astype(float) / X_mar[:, np.newaxis]
    return cond_Xy, cond_yX


def _compute_specific_info(label, y_mar_, cond_Xy, cond_yX, joint):
    Ispec = 0
    n = cond_Xy.shape[0]
    mar = y_mar_[label]

    for x in range(n):
        if mar > 1e-8:
            c = cond_yX[x, label]
            if c > 1e-8:
                contrib = cond_Xy[x, label] * (log2(1.0 / mar) - log2(1.0 / c))
                Ispec += contrib

    return Ispec


def _group_without_unit(group, unit):
    """
    Returns the tuple given by `group without the element give by `uni`.
    """
    if isinstance(unit, int):
        unit = [unit]
    return tuple(k for k in group if not k in unit)


def _joint_sub(X, y, binary=True):
    """
    Computes the joint probability of all groups of `n-1` variables of `X`.
    """
    joints = []
    for i in range(X.shape[1]):
        group = _group_without_unit(range(X.shape[1]), i)
        joints.append(_joint_probability(X[:, group], y, binary=binary))
    return joints
