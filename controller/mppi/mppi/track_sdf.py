import numpy as np
import cv2
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

class TrackSDF:
    def __init__(self, wx, wy, wyaw, w_left, w_right, resolution=0.05, map_padding=2.0):
        """
        Args:
            wx, wy, wyaw, w_left, w_right: Numpy arrays of waypoint data
            resolution: Grid resolution in meters
            map_padding: Extra padding around the track bounds in meters (expands map outward)
        """
        self.resolution = resolution
        self.map_padding = map_padding
        
        # 1. Store Waypoints
        self.wx = wx
        self.wy = wy
        self.wyaw = wyaw
        self.w_left = w_left
        self.w_right = w_right
        
        # 2. Calculate Boundaries (need normals for proper bounds)
        waypoints = np.stack([wx, wy], axis=1)
        gradients = np.gradient(waypoints, axis=0)
        norms = np.linalg.norm(gradients, axis=1, keepdims=True)
        tangents = gradients / (norms + 1e-6)
        normals = np.zeros_like(tangents)
        normals[:, 0] = -tangents[:, 1]
        normals[:, 1] = tangents[:, 0]
        
        left_bound = waypoints + normals * w_left[:, np.newaxis]
        right_bound = waypoints - normals * w_right[:, np.newaxis]
        
        # 3. Define Grid Bounds (include ALL boundaries, not just centerline)
        all_x = np.concatenate([wx, left_bound[:, 0], right_bound[:, 0]])
        all_y = np.concatenate([wy, left_bound[:, 1], right_bound[:, 1]])
        
        self.min_x = np.min(all_x) - map_padding
        self.max_x = np.max(all_x) + map_padding
        self.min_y = np.min(all_y) - map_padding
        self.max_y = np.max(all_y) + map_padding
        
        self.width_cells = int((self.max_x - self.min_x) / resolution)
        self.height_cells = int((self.max_y - self.min_y) / resolution)
        
        self.origin = np.array([self.min_x, self.min_y])
        self.map_size = (self.height_cells, self.width_cells) # H, W order for CV2
        
        print(f"[TrackSDF] Building Map: {self.width_cells}(W) x {self.height_cells}(H) ({resolution}m res)" , flush=True)
        
        # 4. Bake Maps
        self._bake_costmap_cv2()
        self._bake_angle_map_kdtree()
        
    def _bake_costmap_cv2(self):
        """
        Rasterize the track to create a Cost Mask using OpenCV.
        Follows user strategy:
        1. Coordinate Transform (World -> Pixel)
        2. Calculate Boundary Polygons
        3. Fill Poly (Drivable Mask)
        4. Distance Transform (Dist from Center)
        5. Combine into Cost Map
        """
        print("[TrackSDF] Baking CostMap using OpenCV Rasterization...", flush=True)
        
        H, W = self.map_size
        waypoints = np.stack([self.wx, self.wy], axis=1)
        widths = np.stack([self.w_left, self.w_right], axis=1)
        
        # 1. World -> Pixel Transform
        # Note: CV2 uses (x, y) = (col, row). Numpy uses (row, col) = (y, x).
        # We process in (x_world, y_world) which maps to (col_idx, row_idx).
        
        def world_to_pix(pts):
            return ((pts - self.origin) / self.resolution).astype(np.int32)
            
        wp_pix = world_to_pix(waypoints)
        
        # 2. Boundary Calculation
        # Use simple normal vector calculation (User snippet logic)
        gradients = np.gradient(waypoints, axis=0)
        norms = np.linalg.norm(gradients, axis=1, keepdims=True)
        tangents = gradients / (norms + 1e-6)
        
        normals = np.zeros_like(tangents)
        normals[:, 0] = -tangents[:, 1]
        normals[:, 1] = tangents[:, 0]
        
        left_bound = waypoints + normals * widths[:, 0:1]
        right_bound = waypoints - normals * widths[:, 1:2]
        
        left_pix = world_to_pix(left_bound)
        right_pix = world_to_pix(right_bound)
        
        # 3. Create Drivable Mask
        # Instead of single polygon (which fails on self-intersecting tracks),
        # fill segment-by-segment as quads (4 corners per segment)
        drivable_mask = np.zeros((H, W), dtype=np.uint8)
        
        for i in range(len(left_pix) - 1):
            # Quad corners: left[i], left[i+1], right[i+1], right[i]
            quad = np.array([
                left_pix[i],
                left_pix[i + 1],
                right_pix[i + 1],
                right_pix[i]
            ], dtype=np.int32)
            cv2.fillConvexPoly(drivable_mask, quad, 1)
        
        # Centerline Mask (for contour cost)
        center_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.polylines(center_mask, [wp_pix.reshape(-1, 1, 2)], isClosed=False, color=1, thickness=1)
        
        # 4. Distance Transforms
        # Distance from walls (boundary of drivable area)
        dist_from_wall_pix = cv2.distanceTransform(drivable_mask, cv2.DIST_L2, 5)
        dist_from_wall = dist_from_wall_pix * self.resolution
        
        # Distance from centerline (for contour cost)
        dist_from_center_pix = cv2.distanceTransform((1 - center_mask).astype(np.uint8), cv2.DIST_L2, 5)
        dist_from_center = dist_from_center_pix * self.resolution
        
        # 5. Final Cost Map
        # Contour cost: NORMALIZED by track half-width, then squared
        # This makes 0.3m deviation = cost 1.0, 0.6m = cost 4.0, etc.
        track_half_width = 0.50  # [m] Normalization constant (Relaxed: 0.5m dev -> cost 1.0)
        contour_normalized = dist_from_center / track_half_width
        contour_cost = contour_normalized ** 2  # 0 at center, 1 at 0.15m, 4 at 0.3m
        
        # Penalize OUTSIDE extremely hard (if dist > 0.5m)
        # But for now, quadratic is steep enough with Small Width.
        
        # Wall proximity cost: exponential increase near walls
        # k controls steepness: higher k = more aggressive near walls
        k = 5.0  # Steepness parameter
        wall_cost = np.exp(-k * dist_from_wall) * 20.0  # 20 at wall, ~0 at center
        
        # Combined cost = contour + wall
        self.cost_map = contour_cost + wall_cost
        
        # Infinite penalty for non-drivable area (collision)
        self.cost_map[drivable_mask == 0] = 10000.0

        self.cost_map = self.cost_map.T
        
        print(f"[TrackSDF] CostMap Baked. Shape: {self.cost_map.shape} (Expected: {self.width_cells}, {self.height_cells})", flush=True)

    def _bake_angle_map_kdtree(self):
        """
        Use KDTree on Dense points to get smooth Yaw field and S (progress) field.
        We calculate YAW from the dense path geometry to ensure it aligns with the track tangent.
        """
        print("[TrackSDF] Baking Angle Map and S Map (KDTree, Geometrically Derived)..." , flush=True)
        
        # Dense upsampling for smoothness
        dists = np.sqrt(np.sum(np.diff(np.stack([self.wx, self.wy], axis=1), axis=0)**2, axis=1))
        dists = np.concatenate([[0], dists])
        s = np.cumsum(dists)
        total_len = s[-1]
        
        # 2cm resolution for yaw
        s_new = np.arange(0, total_len, 0.02)
        
        from scipy.interpolate import interp1d
        f_x = interp1d(s, self.wx, kind='linear', fill_value="extrapolate")
        f_y = interp1d(s, self.wy, kind='linear', fill_value="extrapolate")
        
        wx_dense = f_x(s_new)
        wy_dense = f_y(s_new)
        
        # Calculate Yaw from Geometry (Gradient)
        dx = np.gradient(wx_dense)
        dy = np.gradient(wy_dense)
        yaw_dense = np.arctan2(dy, dx)
        
        # s_dense is just s_new (arc-length parameter)
        s_dense = s_new
        
        # KDTree
        active_dense = np.stack([wx_dense, wy_dense], axis=1)
        kdtree = KDTree(active_dense)
        
        # Grid points
        x_range = np.linspace(self.min_x, self.max_x, self.width_cells)
        y_range = np.linspace(self.min_y, self.max_y, self.height_cells)
        grid_x, grid_y = np.meshgrid(x_range, y_range, indexing='ij')
        
        points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
        _, indices = kdtree.query(points, k=1)
        
        # Nearest Tangent Yaw (for reference, but we use Flow for control)
        nearest_yaw = yaw_dense[indices]
        self.angle_map = nearest_yaw.reshape(self.width_cells, self.height_cells)
        
        # S Map: arc-length progress at each grid point
        nearest_s = s_dense[indices]
        self.s_map = nearest_s.reshape(self.width_cells, self.height_cells)
        
        # --- Flow Map (Desired Heading) Calculation ---
        # Instead of just using nearest tangent, find a point AHEAD on the track
        # and compute the angle towards it. This bakes Pure Pursuit into the map.
        
        # Lookahead distance on path (e.g., 2.0m)
        lookahead_dist = 1.0  # [m] Lookahead for map-based guidance
        
        # Current index on path (indices per grid point)
        # s_dense resolution is 0.02m. So 1.0m is 50 indices.
        lookahead_idx_offset = int(lookahead_dist / 0.02)
        
        # Find lookahead indices (cyclic for closed track)
        lookahead_indices = (indices + lookahead_idx_offset) % len(wx_dense)
        
        # Target positions for each grid point
        target_wx = wx_dense[lookahead_indices]
        target_wy = wy_dense[lookahead_indices]
        
        # Vector from grid point to target point
        vec_x = target_wx - points[:, 0]
        vec_y = target_wy - points[:, 1]
        
        # Desired Heading (Flow Angle)
        # We store Cos and Sin components to avoid interpolation issues near PI/-PI
        flow_angle = np.arctan2(vec_y, vec_x)
        self.flow_map_x = np.cos(flow_angle).reshape(self.width_cells, self.height_cells)
        self.flow_map_y = np.sin(flow_angle).reshape(self.width_cells, self.height_cells)
        
        print("[TrackSDF] Angle/Flow(Vec)/S Map Baked. Flow lookahead:", lookahead_dist, "m", flush=True)

    def get_jax_maps(self):
        """
        Return JAX-compatible maps and metadata for MPPI solver.
        """
        import jax.numpy as jnp
        
        map_data = {
            'cost_map': jnp.array(self.cost_map),
            'angle_map': jnp.array(self.angle_map),
            'flow_map_x': jnp.array(self.flow_map_x),  # Desired heading vector X (Cos)
            'flow_map_y': jnp.array(self.flow_map_y),  # Desired heading vector Y (Sin)
            's_map': jnp.array(self.s_map)  # Arc-length progress map
        }
        
        map_metadata = {
            'origin': np.array([self.min_x, self.min_y]),
            'resolution': self.resolution,
            'shape': (self.width_cells, self.height_cells),
            'track_length': float(np.max(self.s_map))  # Total track length in meters
        }
        
        return map_data, map_metadata

    def show_debug_map(self):
        """
        Debug visualization.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # 1. Cost Map
        # Note: Transpose for imshow (row, col) = (y, x)
        # self.cost_map is (W, H). imshow needs (H, W) if origin=lower? 
        # Actually imshow expects (Rows, Cols). 
        # Rows = Y axis (Height). Cols = X axis (Width).
        # self.cost_map is (Width, Height). So we transpose to (Height, Width).
        cost_vis = self.cost_map.T
        
        # Clip for visibility
        cost_vis = np.clip(cost_vis, 0, 10.0)
        
        im1 = axes[0].imshow(cost_vis, origin='lower', extent=[self.min_x, self.max_x, self.min_y, self.max_y], cmap='jet')
        axes[0].set_title("Cost Map (Clipped 0-10)")
        fig.colorbar(im1, ax=axes[0])
        axes[0].plot(self.wx, self.wy, 'w--', linewidth=1, alpha=0.7)
        
        # Overlay boundaries
        # Normalize yaw
        yaw = self.wyaw
        nx = -np.sin(yaw)
        ny = np.cos(yaw)
        bx_l = self.wx + nx * self.w_left
        by_l = self.wy + ny * self.w_left
        bx_r = self.wx - nx * self.w_right
        by_r = self.wy - ny * self.w_right
        
        axes[0].plot(bx_l, by_l, 'r-', linewidth=1, label='Left')
        axes[0].plot(bx_r, by_r, 'b-', linewidth=1, alpha=0.7, label='Right')
        axes[0].legend()
        
        # 2. Angle Map
        im2 = axes[1].imshow(self.angle_map.T, origin='lower', extent=[self.min_x, self.max_x, self.min_y, self.max_y], cmap='hsv', vmin=-np.pi, vmax=np.pi)
        axes[1].set_title("Yaw Field")
        
        # Quiver
        skip = max(1, int(self.width_cells / 25))
        x_grid, y_grid = np.meshgrid(np.linspace(self.min_x, self.max_x, self.width_cells),
                                     np.linspace(self.min_y, self.max_y, self.height_cells), indexing='ij')
        q_x = x_grid[::skip, ::skip]
        q_y = y_grid[::skip, ::skip]
        q_yaw = self.angle_map[::skip, ::skip]
        q_u = np.cos(q_yaw)
        q_v = np.sin(q_yaw)
        axes[1].quiver(q_x, q_y, q_u, q_v, color='white', scale=None, scale_units='xy', angles='xy')
        
        plt.tight_layout()
        plt.show(block=True)
        print("[TrackSDF] Visualization updated.", flush=True)
