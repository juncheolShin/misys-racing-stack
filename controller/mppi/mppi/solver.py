import os
import jax
import jax.numpy as jnp
from pathlib import Path
from functools import partial

from .discretizers import rk4_discretization, euler_discretization
from .cost import calculate_cost

# ===== JAX Optimization for GPU Performance =====
jax.config.update("jax_enable_x64", False)
COMPUTE_DTYPE = jnp.bfloat16  # AGX Orin Ampere Tensor Core optimization

jax_cache_dir = Path.home() / "jax_cache"
jax_cache_dir.mkdir(exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(jax_cache_dir))
jax.config.update("jax_default_prng_impl", "threefry2x32")


def truncated_gaussian_sampler(key, mean, low, high, cov):
    """
    Multivariate truncated Gaussian sampler using Cholesky decomposition.
    """
    R = jnp.linalg.cholesky(cov)
    adjusted_low = (low - mean) / jnp.diag(R)
    adjusted_high = (high - mean) / jnp.diag(R)
    samples = jax.random.truncated_normal(
        key, lower=adjusted_low, upper=adjusted_high,
    )
    return mean + R @ samples


class MPPISolver:
    """
    Path-tracking Model Predictive Path Integral (MPPI) controller.
    paper: https://arxiv.org/pdf/1707.02342
    base code: https://github.com/google-research/google-research/tree/master/jax_mpc

    Simplified — single truncated-normal sampling, open-loop rollout.
    Map-based collision cost is retained.
    """

    def __init__(
        self,
        config,
        model,
        discretizer=euler_discretization,
    ) -> None:
        self.config = config
        self.model = model
        self.discretizer = discretizer
        self.control_params = self._init_control()
        self.Q = jnp.array(config.Q)
        self.R = jnp.array(config.R)
        self._cached_P = jnp.array(config.P)
        self.p = jnp.asarray(self.model.parameters_vector_from_config(self.model.params))
        self.nu_eye = jnp.eye(self.config.nu)
        self.nu_zeros = jnp.zeros((self.config.nu,))
        self.key = jax.random.PRNGKey(0)

        # Pre-compute returns triu matrix (constant)
        self._triu_mat = jnp.triu(jnp.ones((config.N, config.N)))

        # Map Data
        self.map_data = None
        self.map_metadata = None

    def set_map(self, map_data, map_metadata):
        """Set the static map data for cost calculation."""
        self.map_data = map_data
        self.map_metadata = map_metadata
        print("[Solver] Map data updated.")

    def _init_control(self):
        a_opt = jnp.zeros((self.config.N, self.config.nu))
        if self.config.adaptive_covariance:
            a_cov = (self.config.u_std ** 2) * jnp.tile(
                jnp.eye(jnp.array(self.config.nu)), (self.config.N, 1, 1)
            )
        else:
            a_cov = None
        return (a_opt, a_cov)

    # ------------------------------------------------------------------
    #  Single-step helpers
    # ------------------------------------------------------------------
    def _step(self, x, u, p):
        return self.discretizer(self.model.f_jax, x, u, p, self.config.dt)

    def _reward(self, x, u, x_ref, Q, R, map_data, map_metadata):
        map_arr = map_data['cost_map']
        cost = calculate_cost(x, u, x_ref, Q, R, map_arr, map_metadata)
        return -cost

    def _returns(self, r):
        return jnp.dot(self._triu_mat, r)

    def _weights(self, R):
        r_range = jnp.maximum(jnp.max(R) - jnp.min(R), 1e-8)
        R_stdzd = (R - jnp.max(R)) / (r_range + self.config.damping)
        w = jnp.exp(R_stdzd / self.config.temperature)
        w = w / jnp.maximum(jnp.sum(w), 1e-8)
        return w

    # ------------------------------------------------------------------
    #  Rollout (open-loop, time-indexed reference)
    # ------------------------------------------------------------------
    @partial(jax.jit, static_argnums=(0,))
    def _rollout(self, u, x0, xref, p, Q, R, map_data, map_metadata):
        """
        Open-loop rollout.
        u:    (N, nu) control sequence
        x0:   (nx,) initial state
        xref: (N, 7) time-indexed reference (one per horizon step)
        """

        def rollout_step(carry, u_t):
            state, step_idx = carry
            u_t = jnp.reshape(u_t, (self.config.nu,))
            state_next = self._step(state, u_t, p)
            r = self._reward(state_next, u_t, xref[step_idx], Q, R,
                             map_data, map_metadata)
            return (state_next, step_idx + 1), (state_next, r)

        _, (s, r) = jax.lax.scan(
            rollout_step, (x0, jnp.int32(0)), u
        )
        return s, r

    # ------------------------------------------------------------------
    #  Single-JIT solve kernel: iteration(s) + warm-start shift
    #  Merges: sampling → rollout → weighting → shift into ONE GPU dispatch.
    # ------------------------------------------------------------------
    @partial(jax.jit, static_argnums=(0,))
    def _solve_jit(self, a_opt, rng, x0, ref_traj,
                   p, Q, R, P, map_data, map_metadata):
        """
        Single JIT: N iterations of MPPI + warm-start shift.
        No adaptive_covariance (a_cov removed for simplicity).
        Returns: (a_opt_shifted, rng_out, a_opt_final, a_last, s_last, r_last)
        """
        # rng split inside JIT — avoids Python-level dispatch
        rng, iter_rng = jax.random.split(rng)

        u_std = jnp.asarray(self.config.u_std, dtype=a_opt.dtype)
        u_std = jnp.maximum(u_std, 1e-6)

        def body_fn(carry, _):
            a_opt_i, rng_i = carry

            rng_da, rng_i = jax.random.split(rng_i)

            # Truncated normal noise centred on a_opt
            adjusted_lower = self.config.u_min - a_opt_i
            adjusted_upper = self.config.u_max - a_opt_i
            scaled_lower = adjusted_lower / u_std
            scaled_upper = adjusted_upper / u_std
            da = jax.random.truncated_normal(
                rng_da,
                lower=scaled_lower,
                upper=scaled_upper,
                shape=(self.config.n_samples, self.config.N, self.config.nu),
            )
            da = da * u_std
            a = a_opt_i + da
            a = jnp.clip(a, self.config.u_min, self.config.u_max)

            # Rollout all samples (open-loop)
            s, r = jax.vmap(
                self._rollout, in_axes=(0, None, None, None, None, None, None, None)
            )(a, x0, ref_traj, p, Q, R, map_data, map_metadata)

            # Importance weighting
            R_traj = jax.vmap(self._returns)(r)
            w = jax.vmap(self._weights, 1, 1)(R_traj)
            da_opt = jax.vmap(jnp.average, (1, None, 1))(da, 0, w)
            a_opt_i = a_opt_i + da_opt

            return (a_opt_i, rng_i), (a, s, r)

        (a_opt_final, _rng), (a_all, s_all, r_all) = jax.lax.scan(
            body_fn, (a_opt, iter_rng),
            None, length=self.config.n_iterations,
        )
        # lax.scan stacks over iterations: take last iteration's outputs
        a_last = a_all[-1]
        s_last = s_all[-1]
        r_last = r_all[-1]

        # Warm-start shift (on GPU)
        a_opt_shifted = jnp.concatenate([a_opt_final[1:], a_opt_final[-1:]], axis=0)

        return a_opt_shifted, rng, a_opt_final, a_last, s_last, r_last

    # ------------------------------------------------------------------
    #  Solve  (minimal Python — single JIT dispatch)
    # ------------------------------------------------------------------
    def solve(self, x0, ref_traj, p=None, Q=None, R=None,
              wheelbase=0.335):
        import time as _time
        _ts0 = _time.time()

        current_Q = Q if Q is not None else self.Q
        current_R = R if R is not None else self.R
        current_p = p if p is not None else self.p
        current_P = self._cached_P
        a_opt = self.control_params[0]

        _ts1 = _time.time()

        # === Single GPU dispatch: rng split + iterations + warm-start ===
        a_opt_shifted, rng_out, a_opt_final, a, s, r = self._solve_jit(
            a_opt, self.key, x0, ref_traj,
            current_p, current_Q, current_R, current_P,
            self.map_data, self.map_metadata,
        )

        _ts2 = _time.time()

        # Update state (all DeviceArrays — no sync)
        self.key = rng_out
        self.control_params = (a_opt_shifted, None)
        self.samples = (a, s, r)

        _ts3 = _time.time()

        # Throttled profiling
        if not hasattr(self, '_solve_profile_counter'):
            self._solve_profile_counter = 0
        self._solve_profile_counter += 1
        if self._solve_profile_counter % 40 == 0:
            # Force synchronization to measure true GPU completion time
            _ts_sync0 = _time.time()
            a_opt_final.block_until_ready()
            _ts_sync1 = _time.time()
            _prep = (_ts1 - _ts0) * 1000.0
            _jit = (_ts2 - _ts1) * 1000.0
            _post = (_ts3 - _ts2) * 1000.0
            _sync = (_ts_sync1 - _ts_sync0) * 1000.0
            print(f"[SolveProfile] prep={_prep:.2f}ms | jit={_jit:.2f}ms "
                  f"| post={_post:.2f}ms | sync={_sync:.2f}ms | total={(_ts3 - _ts0)*1000.0:.2f}ms",
                  flush=True)

        return a_opt_final
