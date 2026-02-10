"""
Utility functions for motion planners

Author: Hongrui Zheng
Last Modified: 5/27/21
"""

import math

import jax
import numpy as np
from numba import njit


import jax.numpy as jnp

def nearest_point(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.

    Args:
        point (jax.numpy.ndarray, (2, )): (x, y) of current pose
        trajectory (jax.numpy.ndarray, (N, 2)): array of (x, y) trajectory waypoints

    Returns:
        nearest_point (jax.numpy.ndarray, (2, )): nearest point on the trajectory to the point
        nearest_dist (float): distance to the nearest point
        t (float): nearest point's location as a segment between 0 and 1 on the vector formed by the closest two points on the trajectory. (p_i---*-------p_i+1)
        i (int): index of nearest point in the array of trajectory waypoints
    """
    # Check if trajectory is empty
    if trajectory.shape[0] == 0:
        return jnp.array([0.0, 0.0]), jnp.inf, 0.0, -1

    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = jnp.sum(diffs**2, axis=1)
    dots = jnp.sum((point - trajectory[:-1, :]) * diffs, axis=1)
    t = dots / jnp.maximum(l2s, 1e-6)  # Avoid division by zero
    t = jnp.clip(t, 0.0, 1.0)
    projections = trajectory[:-1, :] + t[:, None] * diffs
    dists = jnp.linalg.norm(projections - point, axis=1)

    # Check if dists is empty
    if dists.shape[0] == 0:
        return jnp.array([0.0, 0.0]), jnp.inf, 0.0, -1

    min_dist_segment = jnp.argmin(dists)
    return (
        projections[min_dist_segment],
        dists[min_dist_segment],
        t[min_dist_segment],
        min_dist_segment,
    )


def calc_ref_trajectory_indices(x, y, cx, cy, v, dt, N):
    """
    Calcuate the indices of the reference trajectory for the next N steps based on the current velocity and the distance between waypoints in the reference trajectory.

    Args:
        x (float): current x position
        y (float): current y position
        v (float): current velocity
        dt (float): time step
        cx (numpy.ndarray): x positions of the reference trajectory waypoints
        cy (numpy.ndarray): y positions of the reference trajectory waypoints
    """

    # Calculate the distance between waypoints in the reference trajectory
    dl = np.linalg.norm(np.array([cx[1], cy[1]]) - np.array([cx[0], cy[0]]))

    # Find the total number of waypoints in the reference trajectory
    ncourse = len(cx)

    # Find nearest index/setpoint from where the trajectories are calculated
    dists_sq = (cx - x)**2 + (cy - y)**2
    ind = int(np.argmin(dists_sq))

    # based on current velocity, distance traveled on the ref line between time steps
    travel = abs(v) * dt
    dind = travel / dl
    ind_list = int(ind) + np.insert(np.cumsum(np.repeat(dind, N)), 0, 0).round().astype(
        int
    )
    ind_list[ind_list >= ncourse] -= ncourse

    return ind_list


def calc_interpolated_reference_trajectory(
    x, y, yaw, cx, cy, cv, dt, N, reference_trajectory
):
    """
    Calculate the interpolated reference trajectory based on the current position and the reference trajectory waypoints.

    Args:
        x (float): current x position
        y (float): current y position
        yaw (float): current yaw angle (-pi to pi)
        cx (numpy.ndarray): x positions of the reference trajectory waypoints
        cy (numpy.ndarray): y positions of the reference trajectory waypoints
        cv (numpy.ndarray): velocities of the reference trajectory waypoints
        dt (float): time step
        N (int): number of points to interpolate
        reference_trajectory (numpy.ndarray): reference trajectory waypoints

    Returns:
        ref_list (numpy.ndarray): interpolated reference trajectory of shape (N+1, nx) where nx is the number of state variables
    """
    # Calculate the distance between waypoints in the reference trajectory
    dl = np.linalg.norm(np.array([cx[1], cy[1]]) - np.array([cx[0], cy[0]]))
    dl = max(dl, 1e-6) # Avoid division by zero

    # Find the total number of waypoints in the reference trajectory
    ncourse = len(cx)

    # --- Pure NumPy nearest-point search (no JAX, no GPU sync) ---
    # 1. Find nearest waypoint index via squared-distance argmin
    dists_sq = (cx - x)**2 + (cy - y)**2
    ind_current = int(np.argmin(dists_sq))

    # 2. Project onto the nearest segment to get fractional t ∈ [0, 1]
    p0 = np.array([cx[ind_current], cy[ind_current]])
    p1 = np.array([cx[(ind_current + 1) % ncourse], cy[(ind_current + 1) % ncourse]])
    seg = p1 - p0
    seg_len_sq = np.dot(seg, seg)
    if seg_len_sq < 1e-12:
        t_current = 0.0
    else:
        t_current = float(np.clip(np.dot(np.array([x, y]) - p0, seg) / seg_len_sq, 0.0, 1.0))

    # start from the velocity at the current index, calculate next point,
    # interpolate linearly the speed and then use that speed to get next point,
    # Repeat this for N points
    current_speed = (1 - t_current) * cv[ind_current] + t_current * cv[
        (ind_current + 1) % ncourse
    ]
    t_list = np.zeros(N + 1)
    t_list[0] = t_current
    for i in range(1, N + 1):
        t_list[i] = t_list[i - 1] + (current_speed * dt) / dl
        current_speed = (1 - t_list[i]) * cv[ind_current] + t_list[i] * cv[
            (ind_current + 1) % ncourse
        ]

    # Get the indices of the previous point to interpolate with for each point
    ind_list = t_list.astype(int) + ind_current

    # Check if the track is closed (loop)
    # If start and end points are close (< 2.0m), assume closed loop
    is_closed = np.linalg.norm(np.array([cx[0], cy[0]]) - np.array([cx[-1], cy[-1]])) < 2.0

    if is_closed:
        # Modulo the indices to wrap around the reference trajectory
        ind_list = ind_list.astype(int) % ncourse
        next_ind_list = (ind_list + 1) % ncourse
        t_list = t_list % 1.0
    else:
        # For open paths (local waypoints), clamp to valid range [0, ncourse-2]
        ind_list = np.clip(ind_list.astype(int), 0, ncourse - 2)
        next_ind_list = ind_list + 1
        t_list = t_list % 1.0

    # Ensure integer indices for array indexing
    ind_list = ind_list.astype(int)
    next_ind_list = next_ind_list.astype(int)

    # Get the previous point and the next point to interpolate with
    prev_states = reference_trajectory[ind_list, :]
    next_states = reference_trajectory[next_ind_list, :]

    # Interpolate between the previous and next points by (1-t)*ref[i] + t*ref[i+1]
    ref_list = (1 - t_list).reshape(-1, 1) * prev_states + t_list.reshape(
        -1, 1
    ) * next_states
    return ref_list


@njit(cache=True)
def intersect_point(point, radius, trajectory, t=0.0, wrap=False):
    """
    starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.

    Assumes that the first segment passes within a single radius of the point

    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    """
    start_i = int(t)
    start_t = np.float32(t % 1.0)
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory).astype(np.float32)
    for i in range(start_i, trajectory.shape[0] - 1):
        start = trajectory[i, :]
        end = trajectory[i + 1, :] + np.float32(1e-6)
        V = np.ascontiguousarray(end - start)

        a = np.dot(V, V)
        b = np.float32(2.0) * np.dot(V, start - point)
        c = (
            np.dot(start, start)
            + np.dot(point, point)
            - np.float32(2.0) * np.dot(start, point)
            - radius * radius
        )
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (np.float32(2.0) * a)
        t2 = (-b + discriminant) / (np.float32(2.0) * a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0], :]
            end = trajectory[(i + 1) % trajectory.shape[0], :] + np.float32(1e-6)
            V = end - start

            a = np.dot(V, V)
            b = np.float32(2.0) * np.dot(V, start - point)
            c = (
                np.dot(start, start)
                + np.dot(point, point)
                - np.float32(2.0) * np.dot(start, point)
                - radius * radius
            )
            discriminant = b * b - 4 * a * c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (np.float32(2.0) * a)
            t2 = (-b + discriminant) / (np.float32(2.0) * a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t


@njit(cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    waypoint_y = np.dot(
        np.array([np.sin(-pose_theta), np.cos(-pose_theta)]),
        lookahead_point[0:2] - position,
    )
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.0
    radius = 1 / (2.0 * waypoint_y / lookahead_distance**2)
    steering_angle = np.arctan(wheelbase / radius)
    return speed, steering_angle


"""
LQR utilities
"""


@njit(cache=True)
def solve_lqr(A, B, Q, R, tolerance, max_num_iteration):
    """
    Iteratively calculating feedback matrix K

    Args:
        A: matrix_a
        B: matrix_b
        Q: matrix_q
        R: matrix_r_
        tolerance: lqr_eps
        max_num_iteration: max_iteration

    Returns:
        K: feedback matrix
    """

    M = np.zeros((Q.shape[0], R.shape[1]))

    AT = A.T
    BT = B.T
    MT = M.T

    P = Q
    num_iteration = 0
    diff = math.inf

    while num_iteration < max_num_iteration and diff > tolerance:
        num_iteration += 1
        P_next = (
            AT @ P @ A
            - (AT @ P @ B + M) @ np.linalg.pinv(R + BT @ P @ B) @ (BT @ P @ A + MT)
            + Q
        )

        # check the difference between P and P_next
        diff = np.abs(np.max(P_next - P))
        P = P_next

    K = np.linalg.pinv(BT @ P @ B + R) @ (BT @ P @ A + MT)

    return K


@njit(cache=True)
def update_matrix(vehicle_state, state_size, timestep, wheelbase):
    """
    calc A and b matrices of linearized, discrete system.

    Args:
        vehicle_state:
        state_size:
        timestep:
        wheelbase:

    Returns:
        A:
        b:
    """

    # Current vehicle velocity
    v = vehicle_state[3]

    # Initialization of the time discrete A matrix
    matrix_ad_ = np.zeros((state_size, state_size))

    matrix_ad_[0][0] = 1.0
    matrix_ad_[0][1] = timestep
    matrix_ad_[1][2] = v
    matrix_ad_[2][2] = 1.0
    matrix_ad_[2][3] = timestep

    # b = [0.0, 0.0, 0.0, v / L].T
    matrix_bd_ = np.zeros((state_size, 1))  # time discrete b matrix
    matrix_bd_[3][0] = v / wheelbase

    return matrix_ad_, matrix_bd_


"""
Geometry utilities
"""


@njit(cache=True)
def quat_2_rpy(x, y, z, w):
    """
    Converts a quaternion into euler angles (roll, pitch, yaw)

    Args:
        x, y, z, w (float): input quaternion

    Returns:
        r, p, y (float): roll, pitch yaw
    """
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = 1.0 if t2 > 1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return roll, pitch, yaw


@njit(cache=True)
def get_rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.ascontiguousarray(np.array([[c, -s], [s, c]]))


@njit(cache=True)
def pi_2_pi(angle):
    return ((angle + math.pi) % (2.0 * math.pi)) - math.pi


# @njit(cache=True)
def sample_traj(clothoid, npts):
    traj = np.empty((npts, 4))
    for i in range(0, npts):
        s = i * (clothoid.length / max(npts - 1, 1))
        traj[i, 0] = clothoid.X(s)
        traj[i, 1] = clothoid.Y(s)
        traj[i, 2] = clothoid.Theta(s)
        traj[i, 3] = np.sqrt(clothoid.XDD(s) ** 2 + clothoid.YDD(s) ** 2)

    return traj


def map_collision(point, map):
    """
    Returns whether a point is in collision with the map
    """
    pass


def input_acceleration_to_speed(v0, acc, dt):
    """
    Returns the speed after applying acceleration for a given time
    """
    return v0 + acc * dt


def input_steering_speed_to_angle(delta_0, delta_v, dt):
    """
    Returns the steering angle after applying steering velocity for a given time
    """
    return delta_0 + delta_v * dt


def jnp_to_np(jnp_array):
    """
    Converts a jax numpy array to a numpy array
    """
    return np.array(jax.device_get(jnp_array))


def calculate_pp_steering(pose, target_point, wheelbase, lookahead_dist=1.0):
    """
    Pure Pursuit steering control law (JAX-compatible).
    Computes desired steering angle to track a target point.
    
    [최적화] GPU-native calculation - no map lookups, only geometry
    
    Args:
        pose: Current pose [x, y, yaw] (rad)
        target_point: Target waypoint [x, y]
        wheelbase: Vehicle wheelbase L (meters)
        lookahead_dist: Lookahead distance for PP law (default 1.0m)
    
    Returns:
        steering: Desired steering angle (rad), clipped to valid range
    
    Theory:
        Pure Pursuit steering law:
        delta = arctan(2 * L * sin(alpha) / lookahead_dist)
        where:
        - L = wheelbase
        - alpha = heading error = desired_heading - current_yaw
        - lookahead_dist = distance to lookahead point (1m typical)
    """
    import jax.numpy as jnp
    
    # Extract pose
    x_curr = pose[..., 0]
    y_curr = pose[..., 1]
    yaw_curr = pose[..., 2]
    
    # Vector from current position to target
    dx = target_point[..., 0] - x_curr
    dy = target_point[..., 1] - y_curr
    
    # Desired heading (angle to target)
    desired_heading = jnp.arctan2(dy, dx)
    
    # Heading error (normalize to [-pi, pi])
    alpha = desired_heading - yaw_curr
    alpha = jnp.arctan2(jnp.sin(alpha), jnp.cos(alpha))
    
    # Pure Pursuit steering law
    # delta = arctan(2*L*sin(alpha) / lookahead_dist)
    # Clamp lookahead to avoid division issues
    lookahead_safe = jnp.maximum(lookahead_dist, 1e-6)
    steering = jnp.arctan((2.0 * wheelbase * jnp.sin(alpha)) / lookahead_safe)
    
    return steering
