import os
import jax
import jax.numpy as jnp
from pathlib import Path
from functools import partial

from .discretizers import rk4_discretization
from .cost import calculate_cost, get_map_value

jax_cache_dir = Path.home() / "jax_cache"
jax_cache_dir.mkdir(exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(jax_cache_dir))


def truncated_gaussian_sampler(key, mean, low, high, cov):
    """
    Multivariate truncated Gaussian sampler using Cholesky decomposition.
    Generates samples from a truncated Gaussian distribution with given mean, covariance, and bounds.

    Parameters:
      key (jax.random.PRNGKey): Random key for sampling
      mean (numpy.ndarray): Mean of the distribution
      low (numpy.ndarray): Lower bounds for each dimension
      high (numpy.ndarray): Upper bounds for each dimension
      cov (numpy.ndarray): Covariance matrix (optional)
    Returns:
      numpy.ndarray: One sample from the truncated Gaussian distribution

    """
    R = jnp.linalg.cholesky(cov)

    # Adjust the bounds for the truncated normal distribution
    adjusted_low = (low - mean) / jnp.diag(R)
    adjusted_high = (high - mean) / jnp.diag(R)

    # Generate truncated standard normal samples
    samples = jax.random.truncated_normal(
        key,
        lower=adjusted_low,
        upper=adjusted_high,
    )

    # Transform back to original space
    return mean + R @ samples


class MPPISolver:
    """
    Path-tracking Model Predictive Path Integral (MPPI) controller.
    paper: https://arxiv.org/pdf/1707.02342 | base code: https://github.com/google-research/google-research/tree/master/jax_mpc

    Args:
        config (MPPIConfig): MPPI configuration object, contains MPPI costs and constraints
        model (DynamicsModel): dynamics model object, used to compute the state derivative
    """

    def __init__(
        self,
        config,
        model,
        discretizer=rk4_discretization
    ) -> None:
        """
        Initialize the MPPI solver.
        Args:
            config (MPPIConfig): MPPI configuration object, contains MPPI costs and constraints
            model (DynamicsModel): dynamics model object, used to compute the state derivative
            discretizer (function, optional): function to discretize the continuous-time dynamics. Defaults to rk4_discretization.
            step_function (function, optional): function of the form _step(self, x, u, p) to compute the next state given current state and control input. If None, uses the discretizer with model's f_jax
            reward_function (function, optional): function of the form _reward(self, x, u, x_ref, Q, R) to compute the reward given current state, control input, reference state, Q, and R. If None, uses the default quadratic cost
        Returns:
            None
        """
        self.config = config
        self.model = model
        self.discretizer = discretizer
        self.control_params = self._init_control()  # [N, nu]
        self.Q = jnp.array(config.Q)
        self.R = jnp.array(config.R)
        self.p = self.model.parameters_vector_from_config(self.model.params)
        self.nu_eye = jnp.eye(self.config.nu)  # [nu, nu]
        self.nu_zeros = jnp.zeros((self.config.nu,))  # [nu]
        self.key = jax.random.PRNGKey(0)
        
        # Map Data (SDF)
        self.map_data = None
        self.map_metadata = None

    def set_map(self, map_data, map_metadata):
        """
        Set the static map data for cost calculation.
        Args:
           map_data: Dict with jnp arrays (dist_map, angle_map, width_map)
           map_metadata: Dict with origin, resolution, shape
        """
        self.map_data = map_data
        self.map_metadata = map_metadata
        print("[Solver] Map data updated.")

    def _init_control(self):
        """
        Initialize the control parameters for MPPI.

        Returns:
            tuple: (a_opt, a_cov) where a_opt is the optimal control input and a_cov is the covariance matrix.
        """

        a_opt = jnp.zeros((self.config.N, self.config.nu))  # [N, nu]
        # a_cov: [N, nu, nu]
        if self.config.adaptive_covariance:
            # note: should probably store factorized cov,
            # e.g. cholesky, for faster sampling
            a_cov = (self.config.u_std**2) * jnp.tile(
                jnp.eye(jnp.array(self.config.nu)), (self.config.N, 1, 1)
            )
        else:
            a_cov = None
        return (a_opt, a_cov)

        
    # Helper for Gaussian Smoothing
    def _apply_gaussian_smoothing(self, x, sigma=4.0):
        # x: [n_samples, N, nu]
        # We smooth along axis 1 (N)
        
        radius = int(4 * sigma + 0.5)
        k = jnp.arange(-radius, radius + 1)
        kernel = jnp.exp(-0.5 / (sigma**2) * (k**2))
        kernel = kernel / jnp.sum(kernel)
        
        def conv_1d(arr):
            return jnp.convolve(arr, kernel, mode='same')
            
        # Transpose to [n_samples, nu, N] so we can vmap over last axis easily if we wanted, 
        # or just map over 0 and 2.
        # Let's map over samples(0) and controls(2), operating on axis 1.
        
        x_T = jnp.transpose(x, (0, 2, 1)) # [n_samples, nu, N]
        
        smoothed_T = jax.vmap(jax.vmap(conv_1d))(x_T) # [n_samples, nu, N]
        
        return jnp.transpose(smoothed_T, (0, 2, 1)) # [n_samples, N, nu]

    @partial(jax.jit, static_argnums=(0))
    def iteration_step(self, input_, env_state, ref_traj, p, Q, R, P, map_data, map_metadata):
        a_opt, a_cov, rng = input_
        rng_guided, rng_explore, rng = jax.random.split(rng, 3)
        
        # --- Guided Sampling ---
        # State-Dependent Feedback Policy based on Flow Map
        n_guided = int(self.config.n_samples * self.config.guided_ratio)
        
        # [Concept] "Broom-shaped" Trajectories (Steering ONLY)
        # We want steering to fan out (High Amp + Smooth), but Velocity should be stable.
        
        # 1. Define Scales
        # u_std comes from params: [steer_std, vel_std]
        # guided_std_scale comes from params (e.g. 0.7)
        base_std = self.config.u_std * self.config.guided_std_scale
        
        # Apply 3.0x multiplier ONLY to Steering (index 0)
        # Velocity (index 1) keeps 1.0x multiplier to prevent divergence
        scales = jnp.array([3.0, 1.0]) 
        guided_std_broom = base_std * scales
        
        # 2. Generate Noise
        raw_noise_guided = jax.random.normal(rng_guided, shape=(n_guided, self.config.N, self.config.nu)) * guided_std_broom
        
        # 3. Apply Gaussian Filter
        smooth_noise_guided = self._apply_gaussian_smoothing(raw_noise_guided, sigma=4.0)
        
        # Rollout Guided Samples (Closed Loop)
        s_guided, r_guided, a_guided = jax.vmap(self._rollout, in_axes=(0, None, None, None, None, None, None, None, None, None))(
            smooth_noise_guided, env_state, ref_traj, p, Q, R, P, map_data, map_metadata, True # use_feedback_policy=True
        )

        
        # --- Exploration Samples (40%) ---
        n_explore = self.config.n_samples - n_guided
        
        # Explore Noise
        base_std_explore = self.config.u_std * self.config.exploration_std_scale
        explore_std_broom = base_std_explore * scales # [3.0, 1.0] scaling
        
        raw_noise_explore = jax.random.normal(rng_explore, shape=(n_explore, self.config.N, self.config.nu)) * explore_std_broom
        
        # Smooth Exploration Noise
        smooth_noise_explore = self._apply_gaussian_smoothing(raw_noise_explore, sigma=4.0)
        
        # Apply to a_opt
        a_explore = a_opt + smooth_noise_explore
        
        # Clip to limits (Naive clipping might ruin smoothing, but necessary)
        a_explore = jnp.clip(a_explore, self.config.u_min, self.config.u_max)
        
        # Rollout Exploration Samples (Open Loop)
        s_explore, r_explore, a_explore_out = jax.vmap(self._rollout, in_axes=(0, None, None, None, None, None, None, None, None, None))(
            a_explore, env_state, ref_traj, p, Q, R, P, map_data, map_metadata, False # use_feedback_policy=False
        )
        
        # --- Combine Groups ---
        # Note: For guided samples, 'a_guided' IS the realized control.
        # For explore samples, 'a_explore_out' IS 'a_explore'.
        a = jnp.concatenate([a_guided, a_explore_out], axis=0)      # [n_samples, N, nu]
        s = jnp.concatenate([s_guided, s_explore], axis=0)      # [n_samples, N, nx]
        r = jnp.concatenate([r_guided, r_explore], axis=0)      # [n_samples, N]
        
        # Calculate 'da' for covariance update: da = a_sample - a_opt
        da = a - a_opt 
        
        R_traj = jax.vmap(self._returns)(r)  # [n_samples, N]
        w = jax.vmap(self._weights, 1, 1)(R_traj)  # [n_samples, N]
        
        # Update a_opt using weighted average of all samples
        da_opt = jax.vmap(jnp.average, (1, None, 1))(da, 0, w)  # [N, nu]
        a_opt = a_opt + da_opt  # [N, nu]
        
        if self.config.adaptive_covariance:
            a_cov = jax.vmap(jax.vmap(jnp.outer))(da, da)  # [n_samples, N, nu, nu]
            a_cov = jax.vmap(jnp.average, (1, None, 1))(
                a_cov, 0, w
            )  # a_cov: [N, nu, nu]
            a_cov = a_cov + self.nu_eye * 0.00001
            
        return (a_opt, a_cov, rng), (a, s, r)

    def _step(self, x, u, p):
        """
        Single-step state prediction function.
        """
        return self.discretizer(self.model.f_jax, x, u, p, self.config.dt)

    def _reward(self, x, u, x_ref, Q, R, map_data, map_metadata):
        """
        Single-step reward
        """
        cost = calculate_cost(x, u, x_ref, Q, R, map_data, map_metadata)
        return -cost

    def _returns(self, r):
        # r: [N]
        return jnp.dot(jnp.triu(jnp.ones((self.config.N, self.config.N))), r)  # R: [N]

    def _weights(self, R):  # pylint: disable=invalid-name
        # R: [n_samples]
        # Prevent NaN when all samples have equal cost
        r_range = jnp.max(R) - jnp.min(R)
        r_range = jnp.maximum(r_range, 1e-8)  # Prevent division by zero
        R_stdzd = (R - jnp.max(R)) / (r_range + self.config.damping)  # pylint: disable=invalid-name
        w = jnp.exp(R_stdzd / self.config.temperature)  # [n_samples] np.float32
        


        w_sum = jnp.maximum(jnp.sum(w), 1e-8)  # Prevent division by zero
        w = w / w_sum  # [n_samples] np.float32
        return w

    @partial(jax.jit, static_argnums=(0, 10))
    def _rollout(self, u_or_noise, x0, xref, p, Q, R, P, map_data, map_metadata, use_feedback_policy=False):
        """
        Rollout the trajectory.
        If use_feedback_policy is True: u_or_noise is 'noise', and control is calculated dynamically using Flow Map.
        If use_feedback_policy is False: u_or_noise is 'control inputs', standard rollout.
        """
        
        def readout_flow_control(state, noise, map_data, map_metadata):
            curr_x = state[0]
            curr_y = state[1]
            curr_yaw = state[4]
            
            # Lookup Flow Map
            flow_cos = get_map_value(map_data['flow_map_x'], map_metadata, curr_x, curr_y)
            flow_sin = get_map_value(map_data['flow_map_y'], map_metadata, curr_x, curr_y)
            desired_heading = jnp.arctan2(flow_sin, flow_cos)
            
            # Heading Error
            heading_error = desired_heading - curr_yaw
            heading_error = jnp.arctan2(jnp.sin(heading_error), jnp.cos(heading_error))
            
            # Feedback Control (Pure Pursuit-like)
            K_steer = self.config.guided_steer_gain
            delta_nominal = jnp.clip(K_steer * heading_error, self.config.u_min[0], self.config.u_max[0])
            
            # Target Velocity (Constant 3.0 or from map)
            # v_nominal = 6.0
            # [User Request] Sample velocity based on current velocity
            v_nominal = state[3]
            
            u_nominal = jnp.array([delta_nominal, v_nominal])
            
            # Add Noise
            u_final = u_nominal + noise
            return u_final

        def rollout_step(x, input_val):
            # input_val is either 'u' (standard) or 'noise' (feedback)
            state, ind = x
            
            if use_feedback_policy:
                u = readout_flow_control(state, input_val, map_data, map_metadata)
            else:
                u = input_val
                
            u = jnp.reshape(u, (self.config.nu,))
            u = jnp.clip(u, self.config.u_min, self.config.u_max)
            
            # Step Dynamics
            state_next = self._step(state, u, p)
            r = self._reward(state_next, u, xref[:, ind + 1], Q, R, map_data, map_metadata)
            
            x_next = (state_next, ind + 1)
            # We return 'u' as part of output to track applied controls
            return x_next, (state_next, r, u)

        if not self.config.scan:
            scan_output = []
            for t in range(self.config.N):
                x0, output = rollout_step((x0, t), u_or_noise[t, :])
                x0 = x0[0]
                scan_output.append(output)
            s, r, u_applied = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *scan_output)
            # s is already (N, 7)
        else:
            state_and_index_init = (x0, 0)
            _, (s, r, u_applied) = jax.lax.scan(
                rollout_step, state_and_index_init, u_or_noise
            )
            # s is already (N, 7)
       
        return (s, r, u_applied)

    def solve(self, x0, ref_traj, vis=True, p=None, Q=None, R=None):
        """
        Solve the MPPI problem for the given initial state and reference trajectory.
        """
        # Update the parameters of the optimization problem
        current_Q = Q if Q is not None else self.Q
        current_R = R if R is not None else self.R
        current_p = p if p is not None else self.p
        current_P = jnp.array(self.config.P) # Use config P
        current_map_data = self.map_data
        current_map_metadata = self.map_metadata

        # Run MPPI iterations
        self.key, subkey = jax.random.split(self.key)
        rng = subkey
        jax_x0 = jnp.array(x0)
        jax_ref = jnp.array(ref_traj)
        a_opt, a_cov = self.control_params
        # Shift controls for Warm Start
        a_opt_shifted = jnp.concatenate(
            [a_opt[1:, :], jnp.expand_dims(self.nu_zeros, axis=0)]
        )  # [N, nu]
        a_opt = a_opt_shifted # Use shifted as initial guess for this iteration
        if self.config.adaptive_covariance:
            a_cov = jnp.concatenate(
                [
                    a_cov[1:, :],
                    jnp.expand_dims((self.config.u_std**2) * self.nu_eye, axis=0),
                ]
            )
        if not self.config.scan or self.config.n_iterations == 1:
            for _ in range(self.config.n_iterations):
                (a_opt, a_cov, rng), (a_sampled, s_sampled, r_sampled) = (
                    self.iteration_step(
                        (a_opt, a_cov, rng),
                        jax_x0,
                        jax_ref,
                        current_p,
                        current_Q,
                        current_R,
                        current_P,
                        current_map_data,
                        current_map_metadata,
                    )
                )
        else:
            (a_opt, a_cov, rng), (a_sampled, s_sampled, r_sampled) = jax.lax.scan(
                lambda input_, _: self.iteration_step(
                    input_, jax_x0, jax_ref, current_p, current_Q, current_R, current_P, current_map_data, current_map_metadata
                )(a_opt, a_cov, rng),
                None,
                length=self.config.n_iterations,
            )
        # [Warm Start Stabilization]
        # Temporal Smoothing: Blend the new optimal controls with the shifted previous controls.
        # This prevents the trajectory from oscillating wildly ("jumping") between frames.
        # alpha = 0.8 means we trust the new optimization 80%, and keep 20% of the old momentum.
        alpha = 0.8
        a_opt_new = a_opt # This is the updated a_opt from iteration_step
        
        # Recover the shifted initial guess (which was stored in a_opt_shifted at start of function)
        # Wait, 'a_opt' variable was overwritten in the loop. We need to store original shifted version.
        # But we can just use the logic: a_opt_final = alpha * a_opt_new + (1-alpha) * a_opt_shifted
        # Let's verify variable names.
        
        # Refactoring to preserve initial guess
        # See below for full block replacement
        
        a_opt = alpha * a_opt_new + (1.0 - alpha) * a_opt_shifted

        self.control_params, self.samples = (
            (a_opt, a_cov),
            (a_sampled, s_sampled, r_sampled),
        )

        # Get the solved for controls
        self.uk = self.control_params[0]  # [N, nu]

        # Optionally rollout the optimal trajectory for visualization
        # The final trajectory is the OPEN-LOOP rollout of the weighted mean inputs (a_opt)
        if vis:
            self.xk, _, _ = self._rollout(
                self.uk, x0, jax_ref, self.p, self.config.Q, self.config.R, current_P, current_map_data, current_map_metadata, use_feedback_policy=False
            )  # [N, nu]
            self.xk = jnp.concatenate([jnp.expand_dims(x0, axis=0), self.xk], axis=0)

            # Make sure xk and uk are in the right shape
            self.xk = jnp.transpose(self.xk)  # [nx, N+1]
        else:
            self.xk = jnp.zeros((self.config.nx, self.config.N + 1))  # [nx, N+1]
        
        self.uk = jnp.transpose(self.uk)  # [nu, N]
        return self.xk, self.uk
