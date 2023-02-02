import functools

import numpy as np
from sklearn.model_selection import train_test_split


def legal_split(split_fn):
    """
    Wraps a splitting function. It ensures that
    * (pre-split) the number of clients is greater than zero;
    * (pre-split) the number of clients is smaller than the number of samples;
    * (pre-split) the original X and y have the same length;
    * (post-split) the X and y for each client have the same length;
    * (post-split) each y contains at least one non-censored sample;
    * (post-split) each dataset contains at lest 'min_samples' samples;
    * (post-split) each sample has been assigned to one client dataset.

    :param split_fn: split function to be wrapped
    :return: wrapped split function
    """
    @functools.wraps(split_fn)
    def legal_split_fn(
            num_clients: int,
            X: np.ndarray,
            y: np.ndarray,
            min_samples: int = 2,
            **kwargs
    ):

        # preliminary checks
        assert num_clients > 0, f"The number of clients must be greater than zero. Found {num_clients} instead."
        assert num_clients <= len(X), f"The number of clients must be <= the number of samples. " \
                                      f"Found {num_clients}, {len(X)} instead."
        assert len(X) == len(y), f"X and y must have the same length. Found {len(X)} and {len(y)} instead."
        original_size = len(X)
        client_size = 0

        # perform split
        client_data = split_fn(num_clients, X, y, min_samples=min_samples, **kwargs)

        # post-split checks
        for _X, _y in client_data:
            assert len(_X) == len(_y), f"Found a different number of inputs and outputs ({len(_X)}, {len(_y)})."
            assert sum(_y["event"]) > 0, "Each client dataset must contain at least one event."
            assert len(_X) >= min_samples, f"Each client dataset must have at least {min_samples} samples."
            client_size += len(_X)
        assert original_size == client_size, f"The original number of samples ({original_size}) differs from the " \
                                             f"client samples ({client_size})."
        return client_data

    return legal_split_fn


def _extract_min_samples(
        num_clients: int,
        X: np.ndarray,
        y: np.ndarray,
        min_samples: int
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """
    Assigns 'min_samples' samples to each client dataset with stratified sampling on the event indicator. Each client
    dataset must contain at least two samples.

    :param num_clients: the number of clients in the federation
    :param X: the original input dataset with shape (n, d)
    :param y: the original output dataset with shape (n,)
    :param min_samples: the minimum number of samples each client must receive
    :return: a tuple containing (i) X without the samples assigned to the clients (ii) y without the samples assigned to
    the clients (iii) a list of tuples containing the client datasets with the minimum number of samples assigned
    """
    assert min_samples >= 2, f"Each dataset should contain at least two samples. Found {min_samples} instead."
    assert len(X) >= num_clients * min_samples, f"Not enough data for the clients."
    client_data = []
    for j in range(num_clients):
        X, Xj, y, yj = train_test_split(X, y, stratify=y["event"], test_size=min_samples)
        client_data.append((Xj, yj))
        assert len(Xj) >= min_samples, f"Each client dataset must contain at least {min_samples} samples."
    return X, y, client_data


@legal_split
def uniform_split(
        num_clients: int,
        X: np.ndarray,
        y: np.ndarray,
        min_samples: int = 2
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Uniformly splits the original (X, y) survival dataset into a list of federated datasets.

    :param num_clients: the number of clients in the federation
    :param X: the original input dataset with shape (n, d)
    :param y: the original input dataset with shape (n,)
    :param min_samples: the minimum number of samples each client must receive
    :return: a list of datasets containing the original samples uniformly split among the federation clients
    """

    X, y, client_data = _extract_min_samples(num_clients, X, y, min_samples)
    cd_size = [len(X) // num_clients] * num_clients
    while len(X) - sum(cd_size) > 0:
        cd_size[np.random.randint(num_clients)] += 1
    for j in range(num_clients - 1):
        X, _X, y, _y = train_test_split(X, y, test_size=cd_size[j], stratify=y["event"])
        Xj, yj = client_data[j]
        Xj, yj = np.concatenate([Xj, _X]), np.concatenate([yj, _y])
        client_data[j] = (Xj, yj)
    Xj, yj = client_data[-1]
    Xj, yj = np.concatenate([Xj, X]), np.concatenate([yj, y])
    client_data[-1] = (Xj, yj)
    np.random.shuffle(client_data)
    return client_data


@legal_split
def quantity_skewed_split(
        num_clients: int,
        X: np.ndarray,
        y: np.ndarray,
        min_samples: int = 2,
        alpha: float = 1.0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Applies quantity-skewed splitting [1] to the original (X, y) survival dataset to obtain a list of federated
    datasets.

    References:
    [1] Archetti, A., Lomurno, E., Lattari, F., Martin, A., & Matteucci, M. (2023). Heterogeneous Datasets for Federated
        Survival Analysis Simulation. https://arxiv.org/abs/2301.12166

    :param num_clients: the number of clients in the federation
    :param X: the original input dataset with shape (n, d)
    :param y: the original output dataset with shape (n,)
    :param min_samples: the minimum number of samples each client must receive
    :param alpha: distribution similarity parameter alpha > 0
    :return: a list of datasets containing the original samples assigned to the federation clients with quantity-skewed
    splitting
    """

    assert alpha > 0.0, f"Alpha must be greater than 0. Found {alpha} instead."

    # extract the minimum number of samples per client
    X, y, client_data = _extract_min_samples(num_clients, X, y, min_samples)

    # Dirichlet distribution evaluation
    d = np.random.dirichlet([alpha] * num_clients, 1)[0]
    d = np.round(d / d.sum() * len(X))

    # rounding error correction
    conv_error = d.sum() - len(X)
    while conv_error > 0:
        rnd_client = np.random.choice(np.where(d > 0)[0])
        d[rnd_client] -= 1
        conv_error -= 1
    while conv_error < 0:
        rnd_client = np.random.randint(num_clients)
        d[rnd_client] += 1
        conv_error += 1
    assert d.sum() == len(X), f"{d.sum()} != {len(X)}"
    d = d.astype(int)

    # build client datasets
    indices = [i for i in range(len(X))]
    for j in range(num_clients):
        idx = list(np.random.choice(indices, size=d[j], replace=False))
        Xj, yj = client_data[j]
        Xj, yj = np.concatenate([Xj, X[idx]]), np.concatenate([yj, y[idx]])
        client_data[j] = (Xj, yj)
    return client_data


@legal_split
def label_skewed_split(
        num_clients: int,
        X: np.ndarray,
        y: np.ndarray,
        min_samples: int = 2,
        alpha: float = 1.0,
        num_bins: int = 10
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Applies label-skewed splitting [1] to the original (X, y) survival dataset to obtain a list of federated datasets.

    References:
    [1] Archetti, A., Lomurno, E., Lattari, F., Martin, A., & Matteucci, M. (2023). Heterogeneous Datasets for Federated
        Survival Analysis Simulation. https://arxiv.org/abs/2301.12166

    :param num_clients: the number of clients in the federation
    :param X: the original input dataset with shape (n, d)
    :param y: the original output dataset with shape (n,)
    :param min_samples: the minimum number of samples each client must receive
    :param alpha: distribution similarity parameter alpha > 0
    :param num_bins: the number of discretization bins to which the times are assigned
    :return: a list of datasets containing the original samples assigned to the federation clients with label-skewed
    splitting
    """

    assert alpha > 0.0, f"Alpha must be greater than 0. Found {alpha} instead."
    assert num_bins > 0, f"The number of bins must be greater than 0. Found {num_bins} instead."

    # extract a minimum number of samples for each client
    X, y, client_data = _extract_min_samples(num_clients, X, y, min_samples)

    # helper variables
    bins = np.linspace(y["time"].min(), y["time"].max(), num_bins + 1)
    y_classes = np.digitize(y["time"], bins[1:], right=True)
    classes = np.sort(np.unique(y_classes))
    num_samples_per_class = [len(np.where(y_classes == classes[c])[0]) for c in range(len(classes))]

    # Dirichlet distribution evaluation
    d = np.random.dirichlet(np.ones(num_clients) * alpha, len(classes))
    for c in range(len(classes)):
        d[c] = np.round(d[c] / d[c].sum() * num_samples_per_class[c])

        # rounding error correction
        conv_error = d[c].sum() - num_samples_per_class[c]
        while conv_error > 0:
            rnd_client = np.random.choice(np.where(d[c] > 0)[0])
            d[c, rnd_client] -= 1
            conv_error -= 1
        while conv_error < 0:
            rnd_client = np.random.randint(num_clients)
            d[c, rnd_client] += 1
            conv_error += 1

        # check if correction worked
        assert d[c].sum() == num_samples_per_class[c], f"{d[c].sum()} != {num_samples_per_class[c]}"
    d = d.astype(int)

    # build client datasets
    class_idx = [np.where(y_classes == classes[c])[0] for c in range(len(classes))]
    for j in range(num_clients):
        idx = []
        for c in range(len(classes)):
            if d[c, j] > 0:
                idx.extend(list(np.random.choice(class_idx[c], size=d[c, j], replace=False)))
        Xj, yj = client_data[j]
        Xj, yj = np.concatenate([Xj, X[idx]]), np.concatenate([yj, y[idx]])
        client_data[j] = (Xj, yj)
    return client_data
