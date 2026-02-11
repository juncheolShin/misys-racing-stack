from config.dynamics_config import (
    DynamicsConfig,
)
from config.model import DynamicBicycleModel
from .solver import MPPISolver
from .utils import jnp_to_np, calc_interpolated_reference_trajectory
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
        visualize: bool = False,
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
            visualize (bool, optional): If True, return NumPy trajectories for RViz. If False, skip expensive conversions.

        Returns:
            target_speed: float (DeviceArray)
            target_steering: float (DeviceArray)
            opt_traj: np.ndarray [N+1, 7] (Optimal Trajectory) or None
            sampled_trajs: np.ndarray [n_samples, N+1, 7] or None
            ref_traj: None
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

        import time as _time
        _t_plan_start = _time.time()

        x = state[0]
        y = state[1]
        v = state[3]
        yaw = state[4]
        # x0 of shape (nx,)
        x0 = np.array([x, y, state[2], v, yaw, state[5], state[6]])

        N = self.solver.config.N
        dt = self.solver.config.dt

        _t0 = _time.time()

        # --- Convert raw waypoints [x, y, psi, kappa, v, s] to full state reference ---
        wx = self.waypoints[:, 0]
        wy = self.waypoints[:, 1]
        wpsi = self.waypoints[:, 2]
        wkappa = self.waypoints[:, 3]
        wv = self.waypoints[:, 4]

        # Calculate derived states
        L = self.params.WHEELBASE
        wdelta = np.arctan(L * wkappa)
        wyaw_rate = wv * wkappa
        wbeta = np.zeros_like(wv)

        # Construct full reference from incoming waypoint list
        full_ref_waypoints = np.stack([wx, wy, wdelta, wv, wpsi, wyaw_rate, wbeta], axis=1)

        _t1 = _time.time()

        # --- Time-indexed reference: interpolate exactly N+1 points at dt spacing ---
        # Uses velocity-based arc-length interpolation from utils.py
        ref_interp = calc_interpolated_reference_trajectory(
            x, y, yaw,
            wx, wy, wv,
            dt, N,
            full_ref_waypoints,
        )
        # ref_interp shape: (N+1, 7) — includes current position as [0]
        # Solver expects (N, 7) — one ref per horizon step, skip the first (current)
        self.ref_traj = np.asarray(ref_interp[1:N+1]).copy()  # (N, 7)

        _t2 = _time.time()

        # Cache the dynamics parameter vector to avoid recomputing every tick
        if not hasattr(self, '_cached_p') or self._cached_params_id is not params:
            p_np = self.model.parameters_vector_from_config(params) if params is not None else None
            self._cached_p = jnp.asarray(p_np) if p_np is not None else None
            self._cached_params_id = params
            if params is not None:
                self.params = params
        else:
            pass
        p = self._cached_p

        if self.pre_processing_fn is not None:
            x0, self.ref_traj = self.pre_processing_fn(x0, self.ref_traj)

        _t3 = _time.time()

        # Solve MPPI with wheelbase from dynamics params
        wheelbase = self.params.LF + self.params.LR

        # Pre-convert ref_traj to JAX DeviceArray once (avoid repeated jnp.array in solve)
        jax_ref = jnp.asarray(self.ref_traj)
        jax_x0 = jnp.asarray(x0)

        # Pre-convert Q, R to JAX if they are NumPy (cache to avoid per-tick conversion)
        if Q is not None:
            if not hasattr(self, '_jax_Q_cache') or self._jax_Q_src is not Q:
                self._jax_Q_cache = jnp.asarray(Q)
                self._jax_Q_src = Q
            jax_Q = self._jax_Q_cache
        else:
            jax_Q = None

        if R is not None:
            if not hasattr(self, '_jax_R_cache') or self._jax_R_src is not R:
                self._jax_R_cache = jnp.asarray(R)
                self._jax_R_src = R
            jax_R = self._jax_R_cache
        else:
            jax_R = None

        self.u_pred = self.solver.solve(
            jax_x0, jax_ref, p=p, Q=jax_Q, R=jax_R, wheelbase=wheelbase
        )

        # Rollout optimal trajectory only when visualization is requested.
        if visualize:
            current_Q = jax_Q if jax_Q is not None else self.solver.Q
            current_R = jax_R if jax_R is not None else self.solver.R
            current_p = p if p is not None else self.solver.p
            self.x_pred, _ = self.solver._rollout(
                self.u_pred,
                jax_x0,
                jax_ref,
                current_p,
                current_Q,
                current_R,
                self.solver.map_data,
                self.solver.map_metadata,
            )
        else:
            self.x_pred = None

        _t4 = _time.time()

        # ---- Outputs ----
        # Return the first control step as 1-D DeviceArray [steer, speed].
        # Scalar indexing like u_pred[0, 0] triggers GPU sync;
        # slicing u_pred[0] returns a DeviceArray (no sync) and we let
        # the caller's float() be the single sync point.
        u_first = self.u_pred[0]            # shape (nu,), still DeviceArray
        target_steering_raw = u_first[0]    # scalar DeviceArray (lazy, no sync yet)
        target_speed_raw = u_first[1]       # scalar DeviceArray (lazy, no sync yet)

        _t5 = _time.time()

        # Detailed timing (throttled via caller)
        _ref_build = (_t1 - _t0) * 1000.0
        _crop_pad = (_t2 - _t1) * 1000.0
        _cache = (_t3 - _t2) * 1000.0
        _solve = (_t4 - _t3) * 1000.0
        _index = (_t5 - _t4) * 1000.0
        _total_plan = (_t5 - _t_plan_start) * 1000.0
        if not hasattr(self, '_plan_profile_counter'):
            self._plan_profile_counter = 0
        self._plan_profile_counter += 1
        if self._plan_profile_counter % 40 == 0:  # Print every 1 second at 40Hz
            print(f"[PlanProfile] ref_build={_ref_build:.2f}ms | crop_pad={_crop_pad:.2f}ms | cache={_cache:.2f}ms | solve={_solve:.2f}ms | index={_index:.2f}ms | total={_total_plan:.2f}ms", flush=True)

        if not visualize:
            return target_speed_raw, target_steering_raw, None, None, None

        # Convert optimal trajectory to NumPy for visualization/publishing markers.
        x_pred_np = jnp_to_np(self.x_pred)

        # Some solver configurations may return an extra batch dimension.
        # Normalize to x_pred_np shape (N, 7).
        if x_pred_np.ndim == 3:
            x_pred_np = x_pred_np[0]

        # Normalize x_pred to (N, 7)
        if x_pred_np.ndim == 2 and x_pred_np.shape[0] == 7:
            x_pred_np = x_pred_np.T

        # Solver returns states for N steps; visualization expects (N+1, 7) including x0.
        x0_np = np.asarray(x0).reshape(1, -1)
        opt_traj = np.concatenate([x0_np, x_pred_np], axis=0)

        # Sampled trajectories for visualization.
        s_sampled_np = jnp_to_np(self.solver.samples[1]) if self.solver.samples is not None else None
        if s_sampled_np is not None and s_sampled_np.ndim == 3 and s_sampled_np.shape[-1] == 7:
            sampled_trajs = np.concatenate(
                [np.repeat(x0_np[None, ...], s_sampled_np.shape[0], axis=0), s_sampled_np],
                axis=1,
            )
        else:
            sampled_trajs = s_sampled_np

        return target_speed_raw, target_steering_raw, opt_traj, sampled_trajs, None
