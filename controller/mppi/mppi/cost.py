import jax.numpy as jnp
import jax.debug

def interpolate_2d(grid, x_idx, y_idx):
    """
    Bilinear interpolation for 2D grid.
    x_idx, y_idx: float indices
    """
    x0 = jnp.floor(x_idx).astype(int)
    x1 = x0 + 1
    y0 = jnp.floor(y_idx).astype(int)
    y1 = y0 + 1
    
    # Clip to bounds
    h, w = grid.shape
    x0 = jnp.clip(x0, 0, h - 1)
    x1 = jnp.clip(x1, 0, h - 1)
    y0 = jnp.clip(y0, 0, w - 1)
    y1 = jnp.clip(y1, 0, w - 1)
    
    # Values
    Ia = grid[x0, y0]
    Ib = grid[x0, y1]
    Ic = grid[x1, y0]
    Id = grid[x1, y1]
    
    wa = (x1 - x_idx) * (y1 - y_idx)
    wb = (x1 - x_idx) * (y_idx - y0)
    wc = (x_idx - x0) * (y1 - y_idx)
    wd = (x_idx - x0) * (y_idx - y0)
    
    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def get_map_value(map_arr, metadata, x, y):
    origin = metadata['origin']
    res = metadata['resolution']
    
    # Convert metric x,y to grid integer indices
    x_idx = (x - origin[0]) / res
    y_idx = (y - origin[1]) / res
    
    return interpolate_2d(map_arr, x_idx, y_idx)

def calculate_cost(x, u, x_ref, Q, R, map_data=None, map_metadata=None):
    """
    Simplified MPPI Cost Function
    
    Cost = cost_map_value + velocity_tracking + control_cost
    """
    # State Extraction
    x_pos = x[..., 0]
    y_pos = x[..., 1]
    v_current = x[..., 3]
    
    # --- State Cost from CostMap ---
    # cost_map already contains contour + wall + collision costs
    map_cost = 0.0
    if map_data is not None and map_metadata is not None:
        map_cost = get_map_value(map_data['cost_map'], map_metadata, x_pos, y_pos)
    
    # Weight by Q[1,1] (contour weight)
    map_cost = map_cost * Q[1, 1]
    
    # --- Velocity Tracking Cost ---
    # Penalize deviation from target velocity
    target_v = 8.0  # [m/s] Target speed
    velocity_cost = (v_current - target_v)**2 * Q[0, 0]  # Q[0,0] = weight_progress
    
    # Debug: Print average costs (only for mean trajectory hopefully, or random sample)
    # jax.debug.print("MapCost: {m:.2f} | VelCost: {v:.2f}", m=jnp.mean(map_cost), v=jnp.mean(velocity_cost))
    
    # --- Control Input Cost ---
    u_steer = u[..., 0]
    u_vel = u[..., 1]
    
    steer_cost = (u_steer**2) * R[0, 0]
    vel_cost = (u_vel**2) * R[1, 1]
    
    control_cost = steer_cost + vel_cost
    
    return map_cost + velocity_cost + control_cost
