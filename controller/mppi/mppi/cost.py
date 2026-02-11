import jax.numpy as jnp
import jax


@jax.jit
def get_map_value_nearest(map_arr, metadata, x, y):
    """
    Get cost value from map via nearest-neighbor grid lookup.
    [최적화] No bilinear interpolation - direct grid indexing (1 memory access)
    
    Args:
        map_arr: Cost map with shape (W, H) from track_sdf (transposed).
                 Indexing: map_arr[x_idx, y_idx]
        metadata: {'origin_x': x0, 'origin_y': y0, 'resolution': res}
        x, y: World coordinates
    
    Returns:
        Cost value at (x, y)
    """
    # 월드 좌표 -> 맵 인덱스 변환
    idx_x = ((x - metadata['origin_x']) / metadata['resolution']).astype(jnp.int32)
    idx_y = ((y - metadata['origin_y']) / metadata['resolution']).astype(jnp.int32)
    
    # map_arr shape is (W, H) — first axis = x, second axis = y
    size_x, size_y = map_arr.shape
    idx_x = jnp.clip(idx_x, 0, size_x - 1)
    idx_y = jnp.clip(idx_y, 0, size_y - 1)
    
    # 직접 조회: map_arr[x_idx, y_idx] (track_sdf가 .T 해서 이 순서)
    return map_arr[idx_x, idx_y]


@jax.jit
def calculate_cost(x, u, x_ref, Q, R, map_arr, metadata):
    """
    MPPI Cost Function - per-step reference tracking + collision

    NOTE:
      - In rollout we pass a single reference state for the current step.
      - Therefore x_ref shape is (7,) (or (1,7)) not a full (N,7) trajectory.
      - Nearest-point search over the whole trajectory should not be done here.

    Args:
        x: current state [x, y, delta, v, yaw, yaw_rate, beta]
        u: control input [delta, v]
        x_ref: reference state for this step [x, y, delta, v, yaw, yaw_rate, beta]
        Q: state cost matrix (7×7)
        R: control cost matrix (2×2)
        map_arr: Cost map (W, H). Lower is better, higher near walls/outside
        metadata: {'origin_x', 'origin_y', 'resolution'}

    Returns:
        Total cost (scalar)
    """
    # Make sure x_ref is (7,)
    x_ref = jnp.reshape(x_ref, (-1,))

    x_pos = x[..., 0]
    y_pos = x[..., 1]
    v_current = x[..., 3]

    # # Reference for this step
    # x_ref_pos = x_ref[0]
    # y_ref_pos = x_ref[1]
    #v_ref = 8.0
    v_ref = x_ref[3]
    # # === Path Tracking Cost (position error) ===
    # dx = x_pos - x_ref_pos
    # dy = y_pos - y_ref_pos
    # path_tracking_cost = (dx * dx + dy * dy) * Q[1, 1]

    # === Map Cost (distance-based cost map) ===
    map_cost = get_map_value_nearest(map_arr, metadata, x_pos, y_pos) * Q[1, 1]

    # === Velocity Tracking Cost (with 1.0 m/s deadzone) ===
    # Penalize only when below v_ref, or when exceeding by > 2.0 m/s
    v_low = jnp.maximum(v_ref - v_current, 0.0)
    v_high = jnp.maximum(v_current - v_ref - 1.0, 0.0)
    v_err = v_low + v_high
    velocity_cost = (v_err ** 2) * Q[0, 0]

    # === Control Input Cost ===
    u_steer = u[..., 0]
    u_vel = u[..., 1]
    control_cost = (u_steer**2) * R[0, 0] + (u_vel**2) * R[1, 1]

    #return path_tracking_cost + map_cost + velocity_cost + control_cost
    return velocity_cost + map_cost