from config.dynamics_config import (
    DynamicsConfig,
)
from config.model import DynamicBicycleModel
from .solver import MPPISolver
from .utils import calc_interpolated_reference_trajectory , jnp_to_np
from config.config import MPCConfig
import jax.numpy as jnp
import numpy as np


class DynamicMPPIPlanner:
    """
    Convenience class that uses MPPI solver with dynamic bicycle model.

    Args:
        solver (MPPISolver, optional): MPPI solver object, contains MPPI parameters
        model (DynamicsModel, optional): dynamics model object, contains the vehicle dynamics
        params (DynamicsConfig, optional): Vehicle parameters for the dynamic model. If none,
        config (MPCConfig, optional): MPC configuration object, contains MPC costs and constraints
    """

    def __init__(
        self,
        params: DynamicsConfig = None,
        model: DynamicBicycleModel = None,
        config: MPCConfig = None,
        solver: MPPISolver = None
        #pre_processing_fn=None
    ):
        print("Initiailizing Dynamic MPPI Planner", flush=True)
        self.params = params
        self.config = config
        self.model = DynamicBicycleModel(self.params)
        self.solver = MPPISolver(self.config , self.model)
        self.pre_processing_fn = None

        self.waypoints = None
    
    def plan(
        self,
        state,
        waypoints,
        params: DynamicsConfig = None,
        Q: np.ndarray = None,
        R: np.ndarray = None,
    ):
        """
        Args:
            state : vehicle's state. List [x, y, delta, v, yaw, yaw_rate, beta]
            waypoints (numpy.ndarray [N x 5], optional): An array of dynamic waypoints, where each waypoint has
            the format [x, y, delta, velocity, heading]. Overrides the static raceline if provided.
            waypoints (numpy.ndarray [N x 6], optional): An array of raw waypoints [x, y, psi, kappa, v, s].
            The planner will convert this to the full state vector required by the model.
            params (DynamicsConfig, optional): Vehicle parameters for the dynamic model. If none, uses default.
            Q (np.ndarray, optional): State cost matrix. If none, uses default.
            R (np.ndarray, optional): Control input cost matrix. If none, uses default.

        Returns:
            target_speed: float
            target_steering: float
            opt_traj: np.ndarray [N+1, 7] (Optimal Trajectory)
            sampled_trajs: np.ndarray [n_samples, N+1, 7] (Sampled Trajectories)
            ref_traj: np.ndarray [N+1, 7] (Interpolated Reference Trajectory)
        """
        if waypoints is not None:
            if waypoints.shape[1] < 3 or len(waypoints.shape) != 2:
                raise ValueError(
                    "Waypoints need to be a (N x m) numpy array with m >= 3!"
                )
            self.waypoints = waypoints
        else:
            if self.waypoints is None:
                raise ValueError(
                    "Please set waypoints to track during planner instantiation or when calling plan()"
                )

        if Q is not None:
            if Q.shape != (
                self.solver.config.Q.shape[0],
                self.solver.config.Q.shape[1],
            ):
                raise ValueError(
                    f"Q must be of shape {self.solver.config.Q.shape}, got {Q.shape}"
                )

        if R is not None:
            if R.shape != (
                self.solver.config.R.shape[0],
                self.solver.config.R.shape[1],
            ):
                raise ValueError(
                    f"R must be of shape {self.solver.config.R.shape}, got {R.shape}"
                )

        x = state[0]
        y = state[1]
        v = state[3]
        yaw = state[4]
        # x0 of shape (nx,)
        x0 = np.array([x, y, state[2], v, yaw, state[5], state[6]])

        # --- Convert raw waypoints [x, y, psi, kappa, v, s] to full state reference ---
        # DynamicBicycleModel state: [x, y, delta, v, yaw, yaw_rate, slip_angle]
        
        wx = self.waypoints[:, 0]
        wy = self.waypoints[:, 1]
        wpsi = self.waypoints[:, 2]
        wkappa = self.waypoints[:, 3]
        wv = self.waypoints[:, 4]
        ws = self.waypoints[:, 5] # Progress (s), used for MPCC cost later

        # Calculate derived states
        L = self.params.WHEELBASE
        wdelta = np.arctan(L * wkappa)  
        wyaw_rate = wv * wkappa         
        wbeta = np.zeros_like(wv)

        # Construct 7D reference trajectory [N, 7]
        # Order: x, y, delta, v, yaw, yaw_rate, beta
        full_ref_waypoints = np.stack([wx, wy, wdelta, wv, wpsi, wyaw_rate, wbeta], axis=1)

        # Interpolate based on current position
        self.ref_traj = calc_interpolated_reference_trajectory(
            x,
            y,
            yaw,
            wx, 
            wy,
            wv, 
            self.solver.config.dt,
            self.solver.config.N,
            full_ref_waypoints
        ).T.copy()
        p = None
        if params is not None:
            p = self.model.parameters_vector_from_config(params)
            self.params = params

        if self.pre_processing_fn is not None:
            x0, self.ref_traj = self.pre_processing_fn(x0, self.ref_traj)

        # Solve MPPI
        # self.solver.samples contains (a_sampled, s_sampled, r_sampled)
        self.x_pred, self.u_pred = self.solver.solve(x0, self.ref_traj, p=p, Q=Q, R=R, vis=True)
        
        self.x_pred = jnp_to_np(self.x_pred)
        self.u_pred = jnp_to_np(self.u_pred)
        
        # Transpose x_pred to [N+1, 7] for visualization (Solver returns [7, N+1])
        self.x_pred = self.x_pred.T
        
        # Get sampled trajectories for visualization (s_sampled)
        # s_sampled shape: [n_samples, N, nx] -> need to check solver output
        # solver.samples[1] is s_sampled
        s_sampled = jnp_to_np(self.solver.samples[1])
        
        self.local_plan = self.ref_traj[:2].T
        # control_solution stores (x, y) path. x_pred is now [N+1, 7], so take all rows, first 2 cols, then transpose to [2, N+1] if needed or keep as path
        # Usually control_solution is expected as [2, N] or [N, 2]. Let's keep it consistent with previous logic but adapted to transposed x_pred.
        # Previous: x_pred[:2, :] -> [2, N+1]. 
        # Current x_pred is [N+1, 7]. So x_pred[:, :2].T -> [2, N+1]
        self.control_solution = np.array(self.x_pred[:, :2]).T
        
        # Calculate target velocity and steering angle from control inputs (accel, steering_rate)
        # [Revert] Use the first optimal step (Index 1) directly
        # The user will handle delay compensation externally.
        target_speed = self.x_pred[1, 3] # 3 is velocity index
        target_steering = self.x_pred[1, 2] # 2 is steering index

        # Clip control targets to vehicle limits
        target_steering = np.clip(target_steering, self.params.MIN_STEER, self.params.MAX_STEER)
        target_speed = np.clip(target_speed, self.params.MIN_SPEED, self.params.MAX_SPEED)
        
        return target_speed, target_steering, self.x_pred, s_sampled, self.ref_traj.T
