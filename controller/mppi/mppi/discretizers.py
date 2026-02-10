import numpy as np
import scipy.linalg as la
import jax.numpy as jnp


def euler_discretization(func, x, u, p, dt):
    """Euler discretization for a given function."""
    return x + dt * func(x, u, p)


def rk4_discretization(func, x, u, p, dt):
    """
    Runge-Kutta 4th order discretization - JAX compatible.
    
    Args:
        func (callable): JAX-compatible dynamics function f(x, u, p)
        x (jnp.ndarray): state [7,]
        u (jnp.ndarray): control input [2,]
        p (jnp.ndarray): parameters
        dt (float): time step
    
    Returns:
        jnp.ndarray: discretized state [7,]
    """
    # RK4 computation with JAX arrays
    k1 = func(x, u, p)
    k2 = func(x + 0.5 * dt * k1, u, p)
    k3 = func(x + 0.5 * dt * k2, u, p)
    k4 = func(x + dt * k3, u, p)
    
    # JAX-native operations
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def system_matrix_discretization(A, B, dt, method="euler"):
    """Discretize a continuous-time linear system matrix A and input matrix B."""
    if method == "euler":
        Ad = np.eye(A.shape[0]) + dt * A
        Bd = dt * B
    elif method == "exact":
        n = A.shape[0]
        m = B.shape[1]
        M = np.zeros((n + m, n + m))
        M[:n, :n] = A
        M[:n, n:] = B
        exp_M = la.expm(M * dt)
        Ad = exp_M[:n, :n]
        Bd = exp_M[:n, n:]
        return Ad, Bd
    else:
        raise ValueError("Invalid discretization method.")
    return Ad, Bd
