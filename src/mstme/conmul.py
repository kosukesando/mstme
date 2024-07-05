import numpy as np
from scipy.optimize import minimize
from scipy.stats import laplace


class ConmulExtremeEstimator:
    def __init__(self, data: np.ndarray, **kwargs):
        """
        # positional arguments
        - data : NDArray with shape (num_vars, num_samples)
        # kwargs
        - 'rng': Generator
        """
        self._data = data
        self._num_vars = data.shape[0]
        self._num_samples = data.shape[1]
        self._rng = kwargs.get("rng", np.random.default_rng(0))

    def estimate(self, threshold: float, **kwargs):
        """
        # arguments
        - threshold : float
        # kwargs
        - 'N_rep' : int
        """
        N_rep = kwargs.get("N_rep", 100)
        data_rep = self._ndist_replacement(N_rep)
        # Estimate conditional model
        params_uc = []
        for i in range(N_rep):
            _data = data_rep[i]
            params_uc.append(_estimate_params(_data, threshold))
        return params_uc

    def _ndist_replacement(self, N_rep: int) -> list[np.ndarray]:
        # Laplace replacement
        replacement = []
        for i in range(N_rep):
            _idx = self._rng.choice(self._num_samples, size=self._num_samples)
            _data = self._data[:, _idx]
            _rep = np.zeros((self._num_vars, self._num_samples))
            for vi in range(self._num_vars):
                _laplace_sample = laplace.rvs(size=self._num_samples)
                _laplace_sample_sorted = np.sort(_laplace_sample)
                _arg = np.argsort(_data[vi])
                _rep[vi, _arg] = _laplace_sample_sorted
            replacement.append(_rep)
        return replacement

    @property
    def num_vars(self):
        return self._num_vars

    @property
    def num_samples(self):
        return self._num_samples


class ConmulExtremeModel:
    def __init__(self, data: np.ndarray, threshold: float, params, **kwargs):
        self._data = data
        """NDArray (num_vars, num_samples)"""
        self._num_vars = self._data.shape[0]
        self._num_samples = self._data.shape[1]
        self._threshold = threshold
        self._params = params
        """NDArray (num_vars, 4)"""
        self._rng: np.random.Generator = kwargs.get("rng", np.random.default_rng(0))

        residual = []
        eps = 1e-10
        for vi in range(self._num_vars):
            _is_e = self._data[vi] > self._threshold
            _x = self._data[vi, _is_e]  # conditioning(extreme)
            _y = np.delete(self._data[:, _is_e], vi, axis=0)  # conditioned
            _a = self._params[vi, 0]
            _b = self._params[vi, 1]
            _z = (_y - _a * _x) / (_x**_b + eps)
            residual.append(_z)
        self._residual = residual
        """list"""

        _largest_component = self._data.argmax(axis=0)
        _is_me = np.empty((self._num_vars, self._num_samples))
        _is_e = self._data > self._threshold
        _is_e_any = _is_e.any(axis=0)
        for vi in range(self._num_vars):
            _is_me[vi] = np.logical_and(_largest_component == vi, _is_e[vi])
        self._ratio = np.count_nonzero(_is_me, axis=1) / np.count_nonzero(_is_e_any)
        """Ratio of components that are the most extreme"""

    def sample(self, N_sample: int) -> np.ndarray:
        """Returns: NDArray with shape (num_vars, N_sample)"""
        # Sample from model
        _threshold_uniform = laplace.cdf(self._threshold)
        _vi_list = self._rng.choice(self._num_vars, size=N_sample, p=self._ratio)

        sample_full = np.zeros((self._num_vars, N_sample))
        for i, vi in enumerate(_vi_list):
            _a = np.asarray(self._params[vi, 0])
            _b = np.asarray(self._params[vi, 1])
            while True:
                std_gum = laplace.ppf(self._rng.uniform(_threshold_uniform, 1, size=1))
                _z = self._rng.choice(self._residual[vi], axis=1)
                _y_given_x = std_gum * _a + (std_gum**_b) * _z
                if (_y_given_x < std_gum).all():
                    _samples = np.insert(np.asarray(_y_given_x), vi, std_gum)
                    sample_full[:, i] = _samples
                    break
        return sample_full

    def sample_with_body(self, N_sample: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns: tuple of NDArrays with shape (num_vars, N_sample_extreme) and (num_vars, N_sample_nonextreme) respectively"""
        # NOTE: Sample from conmul model + empirical distribution taking the ratio of extreme
        # vs non-extreme into regard
        N_sample_extreme = round(
            (np.count_nonzero(self.is_extreme_any()) / self._num_samples) * N_sample
        )
        N_sample_nonextreme = N_sample - N_sample_extreme
        sample_extreme = self.sample(N_sample_extreme)
        sample_nonextreme = self._rng.choice(
            self._data[:, ~self.is_extreme_any()],
            axis=1,
            size=N_sample_nonextreme,
            replace=True,
        )
        return sample_extreme, sample_nonextreme

    def is_extreme(self):
        return self._data > self._threshold

    def is_extreme_any(self):
        return self.is_extreme().any(axis=0)

    def is_most_extreme(self):
        _stm_g_largest = self._data.argmax(axis=0)
        _is_me = np.empty((self._num_vars, self._num_events))
        for vi in range(self._num_vars):
            _is_me[vi] = np.logical_and(_stm_g_largest == vi, self.is_extreme()[vi])
        return _is_me

    @property
    def data(self):
        """NDArray (num_vars, num_samples)"""
        return self._data

    @property
    def threshold(self):
        """Common threshold"""
        return self._threshold

    @property
    def params(self):
        """Conditional model parameters"""
        return self._params

    @property
    def residual(self):
        """Residual"""
        return self._residual


def _cost_func(p: list, x: np.ndarray, y: np.ndarray) -> float:
    """
    cost(p,data,vi)->float
    p: parameter; [a,b,mu,sigma]
    x: ndarray with shape(num_events,) conditioning
    y: ndarray with shape(num_vars-1, num_events) conditioned
    """
    q = 0
    a = p[0]
    b = p[1]
    mu = p[2]
    sg = p[3]
    if y.ndim < 2:
        y = np.expand_dims(y, axis=0)
    if (x < 0).any():
        raise (ValueError())
    for vj in range(y.shape[0]):
        _qj = np.sum(
            np.log(sg * x**b)
            + 0.5 * ((y[vj] - (a * x + mu * x**b)) / (sg * x**b)) ** 2
        )
        if np.isnan(_qj):
            print(f"Qj is NaN a:{a:0.5f}, {b:0.5f}, {mu:0.5f}, {sg:0.5f}")
            print(f"{x}, {y[vj]}")
        q += _qj
    return q


def _jacobian_custom(p, x, y) -> np.ndarray:
    a = p[0]
    b = p[1]
    mu = p[2]
    sg = p[3]
    da = np.sum(-(x ** (1 - 2 * b) * (-a * x - mu * x**b + y)) / sg**2)
    db = np.sum(
        (
            x ** (-2 * b)
            * np.log(x)
            * (
                -(a**2) * x**2
                + a * x * (2 * y - mu * x**b)
                + sg**3 * x ** (3 * b)
                + mu * y * x**b
                - y**2
            )
        )
        / sg**2
    )
    dm = np.sum(-(x ** (-b) * (-a * x - mu * x**b + y)) / sg**2)
    ds = np.sum(x**b - (x ** (-2 * b) * (a * x + mu * x**b - y) ** 2) / sg**3)
    return np.array([da, db, dm, ds])


def _estimate_params(data: np.ndarray, threshold: float, **kwargs):
    """
    # arguments
    data: np.ndarray (shape:(num_vars, num_samples))
    threshold: float
    """
    num_vars = data.shape[0]
    num_samples = data.shape[1]
    # Estimate conditional model parameters
    # TODO: use kwargs
    lb = [-2, None, -5, 0.1]
    ub = [2, 1, 5, None]
    params = np.zeros((num_vars, 4))
    a0 = np.random.uniform(low=lb[0], high=ub[0])
    b0 = np.random.uniform(low=-1, high=ub[1])
    m0 = np.random.uniform(low=-1, high=1)
    s0 = 1
    _p0 = np.array([a0, b0, m0, s0])

    for vi in range(num_vars):
        _mask = np.logical_and((data[vi, :] > threshold), (~np.isinf(data[vi, :])))
        # NOTE: isinf here to exclude inf data that appears after the probability integral transformation (mainly when the GPD shape param is <-0.5)
        x = data[vi, _mask]  # conditioning
        y = np.delete(data[:, _mask], vi, axis=0)  # conditioned
        optres = minimize(
            _cost_func,
            _p0,
            args=(x, y),
            jac=_jacobian_custom,
            method="L-BFGS-B",
            bounds=(
                (lb[0], ub[0]),
                (lb[1], ub[1]),
                (lb[2], ub[2]),
                (lb[3], ub[3]),
            ),
        )
        _param = optres.x
        params[vi, :] = _param
    return params
