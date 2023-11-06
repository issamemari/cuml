# Copyright (c) 2019-2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import Any, Optional, Tuple, Union

from cuml.common import input_to_cuml_array
from cuml.internals.array import array_to_memory_order
from cuml.internals.input_utils import get_supported_input_type
from cuml.internals.safe_imports import (
    cpu_only_import,
    gpu_only_import,
    gpu_only_import_from,
)

cudf = gpu_only_import("cudf")
cp = gpu_only_import("cupy")
cupyx = gpu_only_import("cupyx")
np = cpu_only_import("numpy")
pd = cpu_only_import("pandas")
cuda = gpu_only_import_from("numba", "cuda")


DEFAULT_TRAIN_SIZE = 0.75


def _stratify_split(X, stratify, labels, n_train, n_test, random_state):
    """
    Function to perform a stratified split based on stratify column.
    Based on scikit-learn stratified split implementation.

    Parameters
    ----------
    X, y: Shuffled input data and labels
    stratify: column to be stratified on.
    n_train: Number of samples in train set
    n_test: number of samples in test set

    Returns
    -------
    X_train, X_test: Data X divided into train and test sets
    y_train, y_test: Labels divided into train and test sets
    """
    x_cudf = False
    labels_cudf = False

    if isinstance(X, cudf.DataFrame):
        x_cudf = True
    elif hasattr(X, "__cuda_array_interface__"):
        X = cp.asarray(X)
    x_order = array_to_memory_order(X)

    # labels and stratify will be only cp arrays
    if isinstance(labels, cudf.Series):
        labels_cudf = True
        labels = labels.values
    elif isinstance(labels, pd.Series):
        labels = labels.values
    elif hasattr(labels, "__cuda_array_interface__"):
        labels = cp.asarray(labels)
    elif isinstance(labels, cudf.DataFrame):
        # ensuring it has just one column
        if labels.shape[1] != 1:
            raise ValueError(
                "Expected one column for labels, but found df"
                "with shape = %d" % (labels.shape)
            )
        labels_cudf = True
        labels = labels[0].values
    elif isinstance(labels, pd.DataFrame):
        # ensuring it has just one column
        if labels.shape[1] != 1:
            raise ValueError(
                "Expected one column for labels, but found df"
                "with shape = %d" % (labels.shape)
            )
        labels = labels[0].values
    elif isinstance(labels, np.ndarray):
        if labels.ndim != 1:
            raise ValueError(
                "Expected one column for labels, but found numpy array"
                "with shape = %d" % (labels.shape)
            )

    labels_order = array_to_memory_order(labels)

    # Converting to cupy array removes the need to add an if-else block
    # for startify column
    if isinstance(stratify, cudf.Series) or isinstance(stratify, pd.Series):
        stratify = stratify.values
    elif hasattr(stratify, "__cuda_array_interface__"):
        stratify = cp.asarray(stratify)
    elif isinstance(stratify, cudf.DataFrame) or isinstance(
        stratify, pd.DataFrame
    ):
        # ensuring it has just one column
        if stratify.shape[1] != 1:
            raise ValueError(
                "Expected one column for stratify, but found df"
                "with shape = %d" % (stratify.shape)
            )
        stratify = stratify[0].values
    elif isinstance(stratify, np.ndarray):
        if stratify.ndim != 1:
            raise ValueError(
                "Expected one column for stratify, but found numpy array"
                "with shape = %d" % (stratify.shape)
            )

    classes, stratify_indices = cp.unique(stratify, return_inverse=True)

    n_classes = classes.shape[0]
    class_counts = cp.bincount(stratify_indices)
    if cp.min(class_counts) < 2:
        raise ValueError(
            "The least populated class in y has only 1"
            " member, which is too few. The minimum"
            " number of groups for any class cannot"
            " be less than 2."
        )

    if n_train < n_classes:
        raise ValueError(
            "The train_size = %d should be greater or "
            "equal to the number of classes = %d" % (n_train, n_classes)
        )

    class_indices = cp.split(
        cp.argsort(stratify_indices), cp.cumsum(class_counts)[:-1].tolist()
    )

    X_train = None

    # random_state won't be None or int, that's handled earlier
    if isinstance(random_state, np.random.RandomState):
        random_state = cp.random.RandomState(seed=random_state.get_state()[1])

    # Break ties
    n_i = _approximate_mode(class_counts, n_train, random_state)
    class_counts_remaining = class_counts - n_i
    t_i = _approximate_mode(class_counts_remaining, n_test, random_state)

    for i in range(n_classes):
        permutation = random_state.permutation(class_counts[i].item())
        perm_indices_class_i = class_indices[i].take(permutation)

        y_train_i = cp.array(
            labels[perm_indices_class_i[: n_i[i]]], order=labels_order
        )
        y_test_i = cp.array(
            labels[perm_indices_class_i[n_i[i] : n_i[i] + t_i[i]]],
            order=labels_order,
        )
        if hasattr(X, "__cuda_array_interface__") or isinstance(
            X, cupyx.scipy.sparse.csr_matrix
        ):
            X_train_i = cp.array(
                X[perm_indices_class_i[: n_i[i]]], order=x_order
            )
            X_test_i = cp.array(
                X[perm_indices_class_i[n_i[i] : n_i[i] + t_i[i]]],
                order=x_order,
            )

            if X_train is None:
                X_train = cp.array(X_train_i, order=x_order)
                y_train = cp.array(y_train_i, order=labels_order)
                X_test = cp.array(X_test_i, order=x_order)
                y_test = cp.array(y_test_i, order=labels_order)
            else:
                X_train = cp.concatenate([X_train, X_train_i], axis=0)
                X_test = cp.concatenate([X_test, X_test_i], axis=0)
                y_train = cp.concatenate([y_train, y_train_i], axis=0)
                y_test = cp.concatenate([y_test, y_test_i], axis=0)

        elif x_cudf:
            X_train_i = X.iloc[perm_indices_class_i[: n_i[i]]]
            X_test_i = X.iloc[perm_indices_class_i[n_i[i] : n_i[i] + t_i[i]]]

            if X_train is None:
                X_train = X_train_i
                y_train = y_train_i
                X_test = X_test_i
                y_test = y_test_i
            else:
                X_train = cudf.concat([X_train, X_train_i], ignore_index=False)
                X_test = cudf.concat([X_test, X_test_i], ignore_index=False)
                y_train = cp.concatenate([y_train, y_train_i], axis=0)
                y_test = cp.concatenate([y_test, y_test_i], axis=0)

    if x_cudf:
        X_train = cudf.DataFrame(X_train)
        X_test = cudf.DataFrame(X_test)

    if labels_cudf:
        y_train = cudf.Series(y_train)
        y_test = cudf.Series(y_test)

    return X_train, X_test, y_train, y_test


def _approximate_mode(class_counts, n_draws, rng):
    """
    CuPy implementataiton based on scikit-learn approximate_mode method.
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/__init__.py#L984

    It is the mostly likely outcome of drawing n_draws many
    samples from the population given by class_counts.

    Parameters
    ----------
    class_counts : ndarray of int
        Population per class.
    n_draws : int
        Number of draws (samples to draw) from the overall population.
    rng : random state
        Used to break ties.

    Returns
    -------
    sampled_classes : cupy array of int
        Number of samples drawn from each class.
        np.sum(sampled_classes) == n_draws
    """
    # this computes a bad approximation to the mode of the
    # multivariate hypergeometric given by class_counts and n_draws
    continuous = n_draws * class_counts / class_counts.sum()
    # floored means we don't overshoot n_samples, but probably undershoot
    floored = cp.floor(continuous)
    # we add samples according to how much "left over" probability
    # they had, until we arrive at n_samples
    need_to_add = int(n_draws - floored.sum())
    if need_to_add > 0:
        remainder = continuous - floored
        values = cp.sort(cp.unique(remainder))[::-1]
        # add according to remainder, but break ties
        # randomly to avoid biases
        for value in values:
            (inds,) = cp.where(remainder == value)
            # if we need_to_add less than what's in inds
            # we draw randomly from them.
            # if we need to add more, we add them all and
            # go to the next value
            add_now = min(len(inds), need_to_add)
            inds = rng.choice(inds, size=add_now, replace=False)
            floored[inds] += 1
            need_to_add -= add_now
            if need_to_add == 0:
                break
    return floored.astype(int)


def _split_numpy(
    X: np.ndarray,
    train_size: int,
    test_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    train = X[:train_size]
    test = X[-test_size:]
    return train, test


def _split_cupy(
    X: Union[cp.ndarray, cupyx.scipy.sparse.csr_matrix],
    train_size: int,
    test_size: int,
    order: str,
) -> Tuple[Union[cp.ndarray, cupyx.scipy.sparse.csr_matrix], ...]:
    train = cp.array(X[:train_size], order=order)
    test = cp.array(X[test_size:], order=order)
    return train, test


def _split_series_or_dataframe(
    X: Union[pd.Series, cudf.Series, pd.DataFrame, cudf.DataFrame],
    train_size: int,
    test_size: int,
):
    train = X.iloc[:train_size]
    test = X.iloc[-test_size:]
    return train, test


def _split_object(
    X: Union[
        pd.Series,
        cudf.Series,
        pd.DataFrame,
        cudf.DataFrame,
        np.ndarray,
        cupyx.scipy.sparse.csr_matrix,
    ],
    train_size: int,
    test_size: int,
):
    if isinstance(X, np.ndarray):
        return _split_numpy(X, train_size, test_size)

    if (
        isinstance(X, pd.Series)
        or isinstance(X, pd.DataFrame)
        or isinstance(X, cudf.Series)
        or isinstance(X, cudf.DataFrame)
    ):
        return _split_series_or_dataframe(X, train_size, test_size)

    if isinstance(X, cupyx.scipy.sparse.csr_matrix) or hasattr(
        X, "__cuda_array_interface__"
    ):
        return _split_cupy(X, train_size, test_size, array_to_memory_order(X))


def _validate_input_type(input_var: Any, input_var_name: str):
    if get_supported_input_type(input_var) is None:
        raise TypeError(
            f"Type of {input_var_name}: {type(input_var)} is not supported. Supported "
            "dtypes: cuDF DataFrame/Series, CuPy array, Numba device array, "
            "NumPy array, and pandas DataFrame/Series"
        )


def _validate_size_parameter(
    size: Union[float, int], name: str, max_value: int
):
    if isinstance(size, float) and not (0 <= size <= 1):
        raise ValueError(f"{name} should be between 0 and 1 (found {size})")
    if isinstance(size, int) and not (0 <= size <= max_value):
        raise ValueError(
            f"{name} should be between 0 and the first dimension of X (found {size})"
        )


def _validate_dimensionality(X, y):
    if y is not None and X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have the same first dimension (found {X.shape[0]} and {y.shape[0]})"
        )


def _determine_train_test_size(
    train_size: Optional[Union[float, int]],
    test_size: Optional[Union[float, int]],
    n_samples: int,
) -> Tuple[int, int]:
    """
    Function to determine train and test sizes in number of samples
    based on the given train_size and test_size.

    If both train_size and test_size are None, defaults to 0.75/0.25
    split.

    If only one of train_size or test_size is None, the other is
    calculated as the complement to the first.

    If both are not None, they are used as is, even if their sum
    exceeds 1 in case of floats or n_samples in case of ints.

    Parameters
    ----------
    train_size: int or float
        Number of samples or proportion of samples to be assigned to
        training set
    test_size: int or float
        Number of samples or proportion of samples to be assigned to
        test set
    n_samples: int
        Total number of samples

    Returns
    -------
    train_size: int
        Number of samples to be assigned to training set
    test_size: int
        Number of samples to be assigned to test set
    """
    if train_size is None and test_size is None:
        train_size = int(n_samples * DEFAULT_TRAIN_SIZE)

    if isinstance(train_size, float):
        train_size = int(n_samples * train_size)

    if test_size is None:
        test_size = n_samples - train_size
    elif isinstance(test_size, float):
        test_size = int(n_samples * test_size)
    elif isinstance(test_size, int) and train_size is None:
        train_size = n_samples - test_size

    return train_size, test_size


def _extract_labels_from_dataframe_column(
    df: Union[pd.DataFrame, cudf.DataFrame], column: str
):
    if not isinstance(df, cudf.DataFrame) and not isinstance(df, pd.DataFrame):
        raise TypeError(
            "X needs to be a cuDF Dataframe or a pandas DataFrame when y is a string"
        )

    if column not in df.columns:
        raise ValueError(
            f"Column name {column} not found in input X dataframe"
        )

    return df.pop(column)


def _split_data(X, y, train_size: int, test_size: int):
    X_train, X_test = _split_object(X, train_size, test_size)
    if y is not None:
        y_train, y_test = _split_object(y, train_size, test_size)

    if y is not None:
        return X_train, X_test, y_train, y_test

    return X_train, X_test


def _shuffle_data(X, y, random_state):
    # Shuffle the data
    if random_state is None or isinstance(random_state, int):
        idxs = cp.arange(X.shape[0])
        random_state = cp.random.RandomState(seed=random_state)
    elif isinstance(random_state, cp.random.RandomState):
        idxs = cp.arange(X.shape[0])
    elif isinstance(random_state, np.random.RandomState):
        idxs = np.arange(X.shape[0])
    else:
        raise TypeError(
            "`random_state` must be an int, NumPy RandomState \
                            or CuPy RandomState."
        )

    random_state.shuffle(idxs)

    if (
        isinstance(X, cudf.DataFrame)
        or isinstance(X, cudf.Series)
        or isinstance(X, pd.DataFrame)
        or isinstance(X, pd.Series)
    ):
        X = X.iloc[idxs]
    elif hasattr(X, "__cuda_array_interface__") or isinstance(
        X, cupyx.scipy.sparse.csr_matrix
    ):
        # numba (and therefore rmm device_array) does not support
        # fancy indexing
        X = cp.asarray(X)[idxs]
    elif isinstance(X, np.ndarray):
        X = X[idxs]

    if (
        isinstance(y, cudf.DataFrame)
        or isinstance(y, cudf.Series)
        or isinstance(y, pd.DataFrame)
        or isinstance(y, pd.Series)
    ):
        y = y.iloc[idxs]
    elif hasattr(y, "__cuda_array_interface__") or isinstance(
        y, cupyx.scipy.sparse.csr_matrix
    ):
        y = cp.asarray(y)[idxs]
    elif isinstance(y, np.ndarray):
        y = y[idxs]

    return X, y, idxs


def train_test_split(
    X,
    y=None,
    test_size: Optional[Union[float, int]] = None,
    train_size: Optional[Union[float, int]] = None,
    shuffle: bool = True,
    random_state: Optional[
        Union[int, cp.random.RandomState, np.random.RandomState]
    ] = None,
    stratify=None,
):
    """
    Partitions a given dataset into four collated objects, mimicking
    Scikit-learn's `train_test_split
    <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html>`_.

    Parameters
    ----------
    X : cudf.DataFrame, pandas.DataFrame, numpy.ndarray, cupyx.scipy.sparse.csr_matrix,
        or cuda_array_interface compliant device array.
        Data to split, has shape (n_samples, n_features)
    y : cudf.DataFrame, cudf.Series, pandas.DataFrame, pandas.Series, numpy.ndarray,
        cupyx.scipy.sparse.csr_matrix, or cuda_array_interface compliant device array,
        optional
        Set of labels for the data, either a series of shape (n_samples) or
        the string label of a column in X (if it is a cuDF/pandas DataFrame)
        containing the labels
    train_size : float or int, optional
        If float, represents the proportion [0, 1] of the data
        to be assigned to the training set. If an int, represents the number
        of instances to be assigned to the training set. Defaults to 0.75
    shuffle : bool, optional
        Whether or not to shuffle inputs before splitting, defaults to True
    random_state : int, CuPy RandomState or NumPy RandomState optional
        If shuffle is true, seeds the generator. Unseeded by default

    stratify: cudf.Series, cuda_array_interface compliant device array,
        numpy.ndarray, or pandas.Series
        optional parameter. When passed, the input is split using this
        as column to startify on. Default=None

    Examples
    --------
    .. code-block:: python

        >>> import cudf
        >>> from cuml.model_selection import train_test_split
        >>> # Generate some sample data
        >>> df = cudf.DataFrame({'x': range(10),
        ...                      'y': [0, 1] * 5})
        >>> print(f'Original data: {df.shape[0]} elements')
        Original data: 10 elements
        >>> # Suppose we want an 80/20 split
        >>> X_train, X_test, y_train, y_test = train_test_split(df, 'y',
        ...                                                     train_size=0.8)
        >>> print(f'X_train: {X_train.shape[0]} elements')
        X_train: 8 elements
        >>> print(f'X_test: {X_test.shape[0]} elements')
        X_test: 2 elements
        >>> print(f'y_train: {y_train.shape[0]} elements')
        y_train: 8 elements
        >>> print(f'y_test: {y_test.shape[0]} elements')
        y_test: 2 elements

        >>> # Alternatively, if our labels are stored separately
        >>> labels = df['y']
        >>> df = df.drop(['y'], axis=1)
        >>> # we can also do
        >>> X_train, X_test, y_train, y_test = train_test_split(df, labels,
        ...                                                     train_size=0.8)

    Returns
    -------

    X_train, X_test, y_train, y_test : cudf.DataFrame or array-like objects
        Partitioned dataframes if X and y were cuDF or pandas.DataFrame objects.
        If `y` was provided as a column name, the column was dropped from `X`.
        Partitioned numba device arrays if X and y were Numba device arrays.
        Partitioned CuPy arrays for any other input.
        Partitioned numpy arrays if X and y were numpy arrays.
    """
    _validate_input_type(X, "X")

    if isinstance(y, str):
        y = _extract_labels_from_dataframe_column(X, y)
    elif y is not None:
        _validate_input_type(y, "y")

    _validate_dimensionality(X, y)

    _validate_size_parameter(train_size, "train_size", X.shape[0])
    _validate_size_parameter(test_size, "test_size", X.shape[0])

    train_size, test_size = _determine_train_test_size(
        train_size, test_size, X.shape[0]
    )

    if shuffle:
        X, y, idxs = _shuffle_data(X, y, random_state)

    if stratify is None:
        return _split_data(X, y, train_size, test_size)

    if (
        isinstance(stratify, cudf.DataFrame)
        or isinstance(stratify, cudf.Series)
        or isinstance(stratify, pd.DataFrame)
        or isinstance(stratify, pd.Series)
    ):
        stratify = stratify.iloc[idxs]
    elif hasattr(stratify, "__cuda_array_interface__") or isinstance(
        stratify, cupyx.scipy.sparse.csr_matrix
    ):
        stratify = cp.asarray(stratify)[idxs]
    elif isinstance(stratify, np.ndarray):
        stratify = stratify[idxs]

    return _stratify_split(
        X,
        stratify,
        y,
        train_size,
        test_size,
        random_state,
    )


class StratifiedKFold:
    """
    A cudf based implementation of Stratified K-Folds cross-validator.

    Provides train/test indices to split data into stratified K folds.
    The percentage of samples for each class are maintained in each
    fold.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    shuffle : boolean, default=False
        Whether to shuffle each class's samples before splitting.
    random_state : int (default=None)
        Random seed

    Examples
    --------
    Splitting X,y into stratified K folds
    .. code-block:: python
        import cupy
        X = cupy.random.rand(12,10)
        y = cupy.arange(12)%4
        kf = StratifiedKFold(n_splits=3)
        for tr,te in kf.split(X,y):
            print(tr, te)
    Output:
    .. code-block:: python
        [ 4  5  6  7  8  9 10 11] [0 1 2 3]
        [ 0  1  2  3  8  9 10 11] [4 5 6 7]
        [0 1 2 3 4 5 6 7] [ 8  9 10 11]

    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        if n_splits < 2 or not isinstance(n_splits, int):
            raise ValueError(
                f"n_splits {n_splits} is not a integer at least 2"
            )

        if random_state is not None and not isinstance(random_state, int):
            raise ValueError(f"random_state {random_state} is not an integer")

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.seed = random_state

    def get_n_splits(self, X=None, y=None):
        return self.n_splits

    def split(self, x, y):
        if len(x) != len(y):
            raise ValueError("Expecting same length of x and y")
        y = input_to_cuml_array(y).array.to_output("cupy")
        if len(cp.unique(y)) < 2:
            raise ValueError("number of unique classes cannot be less than 2")
        df = cudf.DataFrame()
        ids = cp.arange(y.shape[0])

        if self.shuffle:
            cp.random.seed(self.seed)
            cp.random.shuffle(ids)
            y = y[ids]

        df["y"] = y
        df["ids"] = ids
        grpby = df.groupby(["y"])

        dg = grpby.agg({"y": "count"})
        col = dg.columns[0]
        msg = (
            f"n_splits={self.n_splits} cannot be greater "
            + "than the number of members in each class."
        )
        if self.n_splits > dg[col].min():
            raise ValueError(msg)

        def get_order_in_group(y, ids, order):
            for i in range(cuda.threadIdx.x, len(y), cuda.blockDim.x):
                order[i] = i

        got = grpby.apply_grouped(
            get_order_in_group,
            incols=["y", "ids"],
            outcols={"order": "int32"},
            tpb=64,
        )
        got = got.sort_values("ids")

        for i in range(self.n_splits):
            mask = got["order"] % self.n_splits == i
            train = got.loc[~mask, "ids"].values
            test = got.loc[mask, "ids"].values
            if len(test) == 0:
                break
            yield train, test

    def _check_array_shape(self, y):
        if y is None:
            raise ValueError("Expecting 1D array, got None")
        elif hasattr(y, "shape") and len(y.shape) > 1 and y.shape[1] > 1:
            raise ValueError(f"Expecting 1D array, got {y.shape}")
        else:
            pass
