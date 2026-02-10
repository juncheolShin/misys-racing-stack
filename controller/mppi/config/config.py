from dataclasses import dataclass, field
from typing import Callable, List
import numpy as np

@dataclass
class MPCConfig:
    """
    Configuration for the MPC controller. Includes the following parameters:

    Args:
        nx (int): Number of states.
        nu (int): Number of control inputs.
        N (int): Planning horizon for the MPC controller.
        Q (np.ndarray): State cost matrix.
        R (np.ndarray): Control input cost matrix.
        Rd (np.ndarray): Control input derivative cost matrix (action rate cost).
        P (np.ndarray): Terminal cost matrix.
        dt (float): Time discretization interval.
    """

    # Horizon and time step
    N: int
    dt: float

    # Dimensions
    nx: int
    nu: int

    # Cost matrices
    Q: np.ndarray
    R: np.ndarray
    Rd: np.ndarray
    P: np.ndarray

    # Constraints (state and input bounds)
    x_min: np.ndarray = field(default=None)
    x_max: np.ndarray = field(default=None)
    u_min: np.ndarray = field(default=None)
    u_max: np.ndarray = field(default=None)
    ud_min: np.ndarray = field(default=None)
    ud_max: np.ndarray = field(default=None)

    def __post_init__(self):
        # Default x_min, x_max, u_min, u_max to infinities if not provided
        if self.x_min is None:
            self.x_min = -np.inf * np.ones(self.nx)
        if self.x_max is None:
            self.x_max = np.inf * np.ones(self.nx)
        if self.u_min is None:
            self.u_min = -np.inf * np.ones(self.nu)
        if self.u_max is None:
            self.u_max = np.inf * np.ones(self.nu)
        if self.ud_min is None:
            self.ud_min = -np.inf * np.ones(self.nu)
        if self.ud_max is None:
            self.ud_max = np.inf * np.ones(self.nu)

        # Check that the dimensions of nx and nu are consistent with Q, Qf, R
        assert self.Q.shape == (self.nx, self.nx), "Q matrix has incorrect dimensions"
        assert self.R.shape == (self.nu, self.nu), "R matrix has incorrect dimensions"
        assert self.P.shape == (self.nx, self.nx), "P matrix has incorrect dimensions"
        assert self.Rd.shape == (self.nu, self.nu), "Rd matrix has incorrect dimensions"
        assert self.x_min.shape == (self.nx,), "x_min has incorrect dimensions"
        assert self.x_max.shape == (self.nx,), "x_max has incorrect dimensions"
        assert self.u_min.shape == (self.nu,), "u_min has incorrect dimensions"
        assert self.u_max.shape == (self.nu,), "u_max has incorrect dimensions"
        assert self.ud_min.shape == (self.nu,), "ud_min has incorrect dimensions"
        assert self.ud_max.shape == (self.nu,), "ud_max has incorrect dimensions"


@dataclass
class MPPIConfig(MPCConfig):
    """
    Configuration for the MPPI controller, inheriting from MPCConfig and adding MPPI-specific parameters.

    Args:
        n_iterations (int): Number of iterations for the MPPI solver.
        n_samples (int): Number of samples for the MPPI solver.
        temperature (float): Temperature for the MPPI solver.
        damping (float): Damping for the MPPI solver.
        u_std (float): Standard deviation of the control noise.
        scan (bool): Whether to scan the control space.
        adaptive_covariance (bool): Whether to adapt the covariance matrix.
    """

    # MPPI specific parameters
    n_iterations: int = field(default=5)
    n_samples: int = field(default=16)
    temperature: float = field(default=0.01)
    damping: float = field(default=0.001)
    u_std: float = field(default=0.5)  # std of the control noise
    scan: bool = field(default=True)
    adaptive_covariance: bool = field(default=False)
    delay_time: float = field(default=0.05)

    def __post_init__(self):
        super().__post_init__()
        # No additional checks needed for MPPI-specific fields

def dynamic_mppi_config():
    # [x, y, delta, v, yaw, yaw_rate, beta]
    return MPPIConfig(
        nx=7,
        nu=2,
        N=10,
        Q=np.diag([5.0, 5.0, 0.0, 5.0, 0.0, 0.0, 0.0]),
        R=np.diag([0.0, 0.00]),
        Rd=np.diag([0.0, 0.00]),
        P=np.diag([5.0, 5.0, 0.0, 5.0, 0.0, 0.0, 0.0]),
        dt=0.1,
        n_iterations=2,
        n_samples=1024,
        adaptive_covariance=True,
        scan=False,
    )