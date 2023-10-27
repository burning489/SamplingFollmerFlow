import numpy as np
from distribution import get_distribution

_EXAMPLES = {
    1: {
        "name": "GM1d",
        "dimension": 1,
        "weights": np.array([0.25, 0.75]),
        "mean_array": np.array([-2, 2]),
        "var_array": np.array([0.25, 0.25]),
    },
    2: {
        "name": "GM1d",
        "dimension": 1,
        "weights": np.array([0.25, 0.75]),
        "mean_array": np.array([-4, 4]),
        "var_array": np.array([0.25, 0.25]),
    },
    3: {
        "name": "GM1d",
        "dimension": 1,
        "weights": np.array([0.25, 0.75]),
        "mean_array": np.array([-8, 8]),
        "var_array": np.array([0.25, 0.25]),
    },
    4: {
        "name": "GMNd",
        "dimension": 2,
        "weights": np.ones((8,)) / 8,
        "mean_array": [
            [4 * np.sin(2 * (i - 1) * np.pi / 8), 4 *
             np.cos(2 * (i - 1) * np.pi / 8)]
            for i in range(1, 9)
        ],
        "cov_array": 0.03 * np.repeat(np.eye(2)[None, ...], 8, axis=0),
    },
    5: {
        "name": "GMNd",
        "dimension": 2,
        "weights": np.ones((16,)) / 16,
        "mean_array": [
            [8 * np.sin(2 * (i - 1) * np.pi / 16), 8 *
             np.cos(2 * (i - 1) * np.pi / 16)]
            for i in range(1, 17)
        ],
        "cov_array": 0.03 * np.repeat(np.eye(2)[None, ...], 16, axis=0),
    },
    6: {
        "name": "GMNd",
        "dimension": 2,
        "weights": np.ones((16,)) / 16,
        "mean_array": [[i, j] for i in range(-3, 4, 2) for j in range(-3, 4, 2)],
        "cov_array": 0.03 * np.repeat(np.eye(2)[None, ...], 16, axis=0),
    },
    7: {
        "name": "GMNd",
        "dimension": 2,
        "weights": np.ones((16,)) / 16,
        "mean_array": [
            [2 * i, 2 * j] for i in range(-3, 4, 2) for j in range(-3, 4, 2)
        ],
        "cov_array": 0.03 * np.repeat(np.eye(2)[None, ...], 16, axis=0),
    },
    8: {
        "name": "GMNd",
        "dimension": 2,
        "weights": np.ones((25,)) / 25,
        "mean_array": [
            [3 * i, 3 * j] for i in range(-2, 3, 1) for j in range(-2, 3, 1)
        ],
        "cov_array": 0.03 * np.repeat(np.eye(2)[None, ...], 25, axis=0),
    },
    9: {
        "name": "GMNd",
        "dimension": 2,
        "weights": np.ones((49,)) / 49,
        "mean_array": [
            [3 * i, 3 * j] for i in range(-3, 4, 1) for j in range(-3, 4, 1)
        ],
        "cov_array": 0.03 * np.repeat(np.eye(2)[None, ...], 49, axis=0),
    },
    10: {
        "name": "GMNd",
        "dimension": 2,
        "weights": np.ones((4,)) / 4,
        "mean_array": [[x, y] for y in [3., -3.] for x in [3., -3.]],
        "cov_array": [[[1.0, rho], [rho, 1.0]] for rho in [-0.9, 0.9, 0.9, -0.9]]
    },
    11: {
        "name": "GM1d",
        "dimension": 1,
        "weights": np.array([0.2, 0.8]),
        "var_array": 0.25*np.ones((2, )),
        "mean_array": np.array([-1., 1.])
    },
    12: {
        "name": "GMNd",
        "dimension": 2,
        "weights": np.array([0.2, 0.8]),
        "cov_array": 0.25*np.repeat(np.eye(2)[None, ...], 2, axis=0),
        "mean_array": np.hstack([np.array([-1., 1.]).reshape(-1, 1), ]*2)
    },
    13: {
        "name": "GMNd",
        "dimension": 3,
        "weights": np.array([0.2, 0.8]),
        "cov_array": 0.25*np.repeat(np.eye(3)[None, ...], 2, axis=0),
        "mean_array": np.hstack([np.array([-1., 1.]).reshape(-1, 1), ]*3)
    },
    14: {
        "name": "GMNd",
        "dimension": 4,
        "weights": np.array([0.2, 0.8]),
        "cov_array": 0.25*np.repeat(np.eye(4)[None, ...], 2, axis=0),
        "mean_array": np.hstack([np.array([-1., 1.]).reshape(-1, 1), ]*4)
    },
    15: {
        "name": "GMNd",
        "dimension": 5,
        "weights": np.array([0.2, 0.8]),
        "cov_array": 0.25*np.repeat(np.eye(5)[None, ...], 2, axis=0),
        "mean_array": np.hstack([np.array([-1., 1.]).reshape(-1, 1), ]*5)
    },
    16: {
        "name": "GMNd",
        "dimension": 6,
        "weights": np.array([0.2, 0.8]),
        "cov_array": 0.25*np.repeat(np.eye(6)[None, ...], 2, axis=0),
        "mean_array": np.hstack([np.array([-1., 1.]).reshape(-1, 1), ]*6)
    },
    17: {
        "name": "GMNd",
        "dimension": 7,
        "weights": np.array([0.2, 0.8]),
        "cov_array": 0.25*np.repeat(np.eye(7)[None, ...], 2, axis=0),
        "mean_array": np.hstack([np.array([-1., 1.]).reshape(-1, 1), ]*7)
    },
    18: {
        "name": "GMNd",
        "dimension": 8,
        "weights": np.array([0.2, 0.8]),
        "cov_array": 0.25*np.repeat(np.eye(8)[None, ...], 2, axis=0),
        "mean_array": np.hstack([np.array([-1., 1.]).reshape(-1, 1), ]*8)
    },
    19: {
        "name": "GMNd",
        "dimension": 9,
        "weights": np.array([0.2, 0.8]),
        "cov_array": 0.25*np.repeat(np.eye(9)[None, ...], 2, axis=0),
        "mean_array": np.hstack([np.array([-1., 1.]).reshape(-1, 1), ]*9)
    },
    20: {
        "name": "GMNd",
        "dimension": 10,
        "weights": np.array([0.2, 0.8]),
        "cov_array": 0.25*np.repeat(np.eye(10)[None, ...], 2, axis=0),
        "mean_array": np.hstack([np.array([-1., 1.]).reshape(-1, 1), ]*10)
    },
}


def register_example(cls=None, *, name=None):
    """A decorator for registering examples."""
    def _register(cls):
        local_name = cls.__name__ if name is None else name
        if local_name in _EXAMPLES:
            raise ValueError(
                f"Already registered class with name: {local_name}")
        _EXAMPLES[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_example(id):
    example_cfg = _EXAMPLES[id]
    return get_distribution(example_cfg["name"])(**example_cfg)
