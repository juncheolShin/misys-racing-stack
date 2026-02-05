import numpy as np
import scipy.linalg as la


def euler_discretization(func, x, u, p, dt):
    """
    Euler discretization for a given function.

    Args:
        func (callable): function to discretize
        x (np.ndarray): state
        u (np.ndarray): control input
        p (np.ndarray): parameters for the function
        dt (float): time step

    Returns:
        np.ndarray: discretized state
    """
    return x + dt * func(x, u, p)


def rk4_discretization(func, x, u, p, dt):
    """
    Runge-Kutta 4th order discretization for a given function.

    Args:
        func (callable): function to discretize
        x (np.ndarray): state
        u (np.ndarray): control input
        p (np.ndarray): parameters for the function
        dt (float): time step

    Returns:
        np.ndarray: discretized state
    """
    k1 = func(x, u, p)
    k2 = func(x + 0.5 * dt * k1, u, p)
    k3 = func(x + 0.5 * dt * k2, u, p)
    k4 = func(x + dt * k3, u, p)
    return x + ((dt * (k1 + 2 * k2 + 2 * k3 + k4)) / 6)


def system_matrix_discretization(A, B, dt, method="euler"):
    """
    Discretize a continuous-time linear system matrix A and input matrix B. For systems with linearization residual C, calculate the residual Cd = f(x,u) - Ad*x - Bd*u.

    Reference:
        - Exact Discretization using No-Input State-Space form: https://ethz.ch/content/dam/ethz/special-interest/mavt/dynamic-systems-n-control/idsc-dam/Lectures/Signals-and-Systems/Lectures/Fall2018/Lecture1_SigSys18.pdf
    Args:
        A (np.ndarray): system matrix
        B (np.ndarray): input matrix
        dt (float): time step
        method (str): discretization method ('euler' or 'exact')

    Returns:
        np.ndarray: discretized system matrix
        np.ndarray: discretized input matrix
    """
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
