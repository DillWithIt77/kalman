import numpy as np
import xarray as xr


class L96Model:
    """Lorenz 96 model implementation"""

    def __init__(self, N=40, F=8.0, dt=0.01):
        self.N = N
        self.F = F
        self.dt = dt
        self.x = np.zeros(N)
        self.t = 0.0

    def rhs(self, x):
        """Right-hand side of the Lorenz-96 model"""
        N = self.N
        dxdt = np.zeros(N)
        for i in range(N):
            dxdt[i] = (x[(i + 1) % N] - x[(i - 2) % N]) * x[(i - 1) % N] - x[i] + self.F
        return dxdt

    def step_forward(self):
        """RK4 time stepping"""
        x = self.x
        dt = self.dt
        k1 = self.rhs(x)
        k2 = self.rhs(x + 0.5 * dt * k1)
        k3 = self.rhs(x + 0.5 * dt * k2)
        k4 = self.rhs(x + dt * k3)
        self.x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.t += dt

    def to_dataset(self):
        return xr.Dataset({'x': (['dim'], self.x), 't': self.t})


class QG1Model:
    """
    Stub for a 1-layer quasi-geostrophic model.

    To implement, this class needs to provide the same interface as Lorenz96Model:
        - self.N     : total number of state variables (e.g. nx * ny for a 2D grid)
        - self.dt    : model timestep
        - self.x     : 1D state vector of length N (flattened from 2D grid if needed)
        - self.t     : current model time
        - step_forward() : advance the model by one timestep

    Suggested libraries: pyqg, or a custom pseudo-spectral implementation.

    Example skeleton:
        def __init__(self, nx=64, ny=64, dt=3600.0, **kwargs):
            self.nx = nx
            self.ny = ny
            self.N  = nx * ny        # total state size
            self.dt = dt
            self.t  = 0.0
            self.x  = np.zeros(self.N)
            # initialize your QG solver here

        def step_forward(self):
            # advance QG model by one timestep
            # update self.x and self.t
            raise NotImplementedError

        def rhs(self, x):
            # optional, only needed if using RK4 externally
            raise NotImplementedError
    """

    def __init__(self, **kwargs):
        raise NotImplementedError("QGModel is not yet implemented. See class docstring for guidance.")

    def step_forward(self):
        raise NotImplementedError

    def rhs(self, x):
        raise NotImplementedError


class Ensemble:
    """
    Generic ensemble of any model that implements the step_forward / x / t / N / dt interface.
    Works with Lorenz96Model, QGModel, or any future model class.
    """

    def __init__(self, models):
        for m in models:
            m.x = np.array(m.x, copy=True)
        self.models = models
        self.nens = len(models)

    def step_forward(self):
        for m in self.models:
            m.step_forward()

    def run_for_steps(self, steps, save_every=1):
        results = []
        for i in range(steps):
            self.step_forward()
            if (i + 1) % save_every == 0:
                x_all = np.array([m.x for m in self.models])
                results.append(x_all)
        x_array = np.array(results)
        times = np.arange(len(results)) * save_every * self.models[0].dt
        return xr.DataArray(
            x_array,
            dims=['time', 'model', 'dim'],
            coords={
                'time': times,
                'model': range(self.nens),
                'dim': range(self.models[0].N)
            }
        )